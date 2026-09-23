from collections.abc import Sequence
import logging
import pathlib
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


class Policy(BasePolicy):

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
        self._model = model
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        if hasattr(model, "write_memory"):
            self._write_memory = nnx_utils.module_jit(model.write_memory)
        if hasattr(model, "get_vision_encoding"):
            self._get_vision_encoding = nnx_utils.module_jit(model.get_vision_encoding)
        self._uses_anchor = getattr(getattr(model, "config", None), "use_anchor_memory", False)
        self._uses_first_frame = getattr(getattr(model, "config", None), "use_first_frame", False)
        if self._uses_first_frame:
            self._first_frame_state_fn = nnx_utils.module_jit(model.first_frame_state)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._memory_state = None
        self._anchor_state = None
        self._first_frame_state = None
        self._prev_h_states = None
        self._prev_actions = None

    def reset(self):
        self._memory_state = None
        # Dropped with the memory state: the anchor and the stored first frame
        # are both re-taken from the first frame of the next episode.
        self._anchor_state = None
        self._first_frame_state = None
        self._prev_h_states = None
        self._prev_actions = None

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        sample_rng, self._rng = jax.random.split(self._rng)

        # Check if the model has episodic memory
        if hasattr(self._model, "write_memory"):
            config = self._model.config
            if self._memory_state is None:
                # Initialize memory state based on configuration
                batch_size = inputs["state"].shape[0]
                depth = len(self._model.read_proj_q)
                state_horizon = 1 if config.memory_value_type in ("vision", "vision_action") else config.action_horizon
                if config.memory_type == "lstm_cell":
                    S_init = jnp.broadcast_to(self._model.S_init_param.value[:, None], (depth, batch_size, state_horizon, config.memory_rank, config.memory_rank))
                    C_init = jnp.zeros((depth, batch_size, config.memory_rank, config.memory_rank), dtype=jnp.float32)
                    self._memory_state = (S_init, C_init)
                else:
                    self._memory_state = jnp.broadcast_to(
                        self._model.S_init_param.value[:, None],
                        (depth, batch_size, state_horizon, config.memory_rank, config.memory_rank)
                    )
            
            # Causal delayed memory writing for vision value inputs:
            # We write the memory using the previous step's hidden states, the current observation's vision encoding, and previous step's actions.
            if self._model.config.memory_value_type in ("vision", "vision_action"):
                if self._prev_h_states is not None:
                    current_vision_encoding = self._get_vision_encoding(_model.Observation.from_dict(inputs))
                    if self._model.config.memory_value_type == "vision":
                        val_concat = current_vision_encoding
                    else:
                        # Sum delta actions along the horizon axis to get net displacement
                        sum_actions = jnp.sum(self._prev_actions, axis=1)  # shape: (B, action_dim)
                        val_concat = jnp.concatenate([current_vision_encoding, sum_actions], axis=-1)  # shape: (B, vision_dim + action_dim)
                    val_concat = jnp.expand_dims(val_concat, axis=1)  # shape: (B, 1, val_concat.shape[-1])
                    self._memory_state = self._write_memory(self._prev_h_states, val_concat, self._memory_state)

            # The episode's first frame, captured once and handed back unchanged
            # on every later call. Preprocessing is deterministic at inference,
            # so these are bit-identical to the images sample_actions builds its
            # current-frame prefix from on this same observation.
            if self._uses_first_frame and self._first_frame_state is None:
                self._first_frame_state = self._first_frame_state_fn(
                    _model.Observation.from_dict(inputs)
                )

            # The model captures z_anchor from the same first-frame image tokens
            # used by its prefix and returns it. This avoids a second SigLIP
            # encode at episode start; reset() makes the next call capture anew.
            sample_result = self._sample_actions(
                sample_rng,
                _model.Observation.from_dict(inputs),
                memory_state=self._memory_state,
                **({"anchor_state": self._anchor_state} if self._uses_anchor else {}),
                **({"first_frame_state": self._first_frame_state} if self._uses_first_frame else {}),
                **self._sample_kwargs
            )
            if self._uses_anchor:
                actions, h_states, self._anchor_state = sample_result
            else:
                actions, h_states = sample_result
            
            if self._model.config.memory_value_type in ("vision", "vision_action"):
                self._prev_h_states = h_states
                self._prev_actions = actions
            else:
                self._memory_state = self._write_memory(h_states, actions, self._memory_state)
        else:
            actions = self._sample_actions(
                sample_rng,
                _model.Observation.from_dict(inputs),
                **self._sample_kwargs
            )

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)

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

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
