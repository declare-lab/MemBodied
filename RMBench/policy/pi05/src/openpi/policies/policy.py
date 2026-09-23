from collections.abc import Sequence
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _tree_rms_delta(before: Any, after: Any) -> at.Array:
    before_leaves = jax.tree.leaves(before)
    after_leaves = jax.tree.leaves(after)
    squared_sum = sum(
        jnp.sum(jnp.square(new.astype(jnp.float32) - old.astype(jnp.float32)))
        for old, new in zip(before_leaves, after_leaves, strict=True)
    )
    element_count = sum(leaf.size for leaf in before_leaves)
    return jnp.sqrt(squared_sum / max(element_count, 1))


class Policy(BasePolicy):
    """Pi0.5 inference with ten interleaved memory streams for five-step replanning.

    Each stream advances once per ten calls (50 environment steps). As in the
    pi0 five-step policy, delayed writes use that stream's full predicted action
    chunk. The episode anchor is shared across streams.
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._rng = rng or jax.random.key(0)
        if hasattr(model, "write_memory"):
            self._write_memory = nnx_utils.module_jit(model.write_memory)
            self._get_vision_encoding = nnx_utils.module_jit(model.get_vision_encoding)
            self._memory_key_diagnostics = nnx_utils.module_jit(model.memory_key_diagnostics)

        self._uses_vector_memory = hasattr(model, "write_memory")
        self._uses_anchor = bool(getattr(getattr(model, "config", None), "use_anchor_memory", False))
        self._collect_memory_diagnostics = os.getenv("OPENPI_LOG_MEMORY_DIAGNOSTICS", "0") == "1"
        self.reset()

    def reset(self) -> None:
        """Drop all episode-local state before the next inference episode."""
        self._memory_state = None
        self._anchor_state = None
        self._prev_h_states = None
        self._prev_actions = None
        self._prev_key_h_states = None
        self._memory_states = [None] * 10
        self._prev_h_states_list = [None] * 10
        self._prev_actions_list = [None] * 10
        self._prev_key_h_states_list = [None] * 10
        self._infer_counter = 0

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        if self._uses_vector_memory:
            slot_idx = self._infer_counter % 10
            self._memory_state = self._memory_states[slot_idx]
            self._prev_h_states = self._prev_h_states_list[slot_idx]
            self._prev_actions = self._prev_actions_list[slot_idx]
            self._prev_key_h_states = self._prev_key_h_states_list[slot_idx]

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        memory_diagnostics = None
        memory_state_delta = jnp.zeros((), dtype=jnp.float32)
        if self._uses_vector_memory:
            if self._memory_state is None:
                self._memory_state = self._model.init_memory_state(inputs["state"].shape[0])

            config = self._model.config
            if (
                config.memory_value_type in ("vision", "vision_action")
                and self._prev_h_states is not None
            ):
                current_vision = self._get_vision_encoding(observation)
                if config.memory_value_type == "vision":
                    memory_value = current_vision
                else:
                    action_delta = jnp.sum(self._prev_actions, axis=1)
                    memory_value = jnp.concatenate([current_vision, action_delta], axis=-1)
                if self._collect_memory_diagnostics:
                    previous_memory_state = self._memory_state
                self._memory_state = self._write_memory(
                    self._prev_h_states, memory_value[:, None, :], self._memory_state
                )
                if self._collect_memory_diagnostics:
                    memory_state_delta = _tree_rms_delta(previous_memory_state, self._memory_state)

            sample_result = self._sample_actions(
                sample_rng,
                observation,
                memory_state=self._memory_state,
                **({"anchor_state": self._anchor_state} if self._uses_anchor else {}),
                **sample_kwargs,
            )
            if self._uses_anchor:
                actions, hidden_states, self._anchor_state, read_metrics = sample_result
            else:
                actions, hidden_states, read_metrics = sample_result

            if self._collect_memory_diagnostics:
                memory_diagnostics = self._memory_key_diagnostics(
                    hidden_states, self._prev_key_h_states
                )
                memory_diagnostics.update(
                    {
                        "memory_gate_mean": read_metrics[0],
                        "read_vector_rms": read_metrics[1],
                        "injected_memory_rms": read_metrics[2],
                    }
                )
                self._prev_key_h_states = hidden_states

            if config.memory_value_type in ("vision", "vision_action"):
                self._prev_h_states = hidden_states
                self._prev_actions = actions
            else:
                if self._collect_memory_diagnostics:
                    previous_memory_state = self._memory_state
                self._memory_state = self._write_memory(hidden_states, actions, self._memory_state)
                if self._collect_memory_diagnostics:
                    memory_state_delta = _tree_rms_delta(previous_memory_state, self._memory_state)
        else:
            actions = self._sample_actions(sample_rng, observation, **sample_kwargs)

        if self._uses_vector_memory:
            self._memory_states[slot_idx] = self._memory_state
            self._prev_h_states_list[slot_idx] = self._prev_h_states
            self._prev_actions_list[slot_idx] = self._prev_actions
            self._prev_key_h_states_list[slot_idx] = self._prev_key_h_states
            self._infer_counter += 1

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if memory_diagnostics is not None:
            outputs["memory_diagnostics"] = {
                key: float(np.asarray(value)) for key, value in memory_diagnostics.items()
            }
            outputs["memory_diagnostics"]["memory_state_delta_rms"] = float(
                np.asarray(memory_state_delta)
            )
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
