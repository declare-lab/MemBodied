import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):

    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):

    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA and memory/delta/anchor weights.
        return _merge_params(loaded_params, params, missing_regex=".*(lora|memory|delta|anchor_|read_proj|write_proj|readout_attn|lstm_forget|unroll_pos_emb|unroll_proj_v|pooling_query|pooling_proj|S_init_param|action_trajectory_proj).*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz",
            gs={"token": "anon"},
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _flatten_dict(d, prefix=(), sep="/"):
    flat = {}
    for k, v in d.items():
        path = (*prefix, str(k))
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, path, sep))
        else:
            flat[sep.join(path)] = v
    return flat


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = _flatten_dict(params, sep="/")
    flat_loaded = _flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref = flat_ref[k]
            loaded_value = v
            # Anchor-concat widens the existing action/time affine map from
            # [2*width, width] to [3*width, width]. Preserve the checkpoint's
            # complete pretrained slice and append the reference model's
            # zero-initialized anchor slice. This is intentionally the only
            # shape-changing load supported here.
            if (
                k.endswith("action_time_mlp_in/kernel")
                and loaded_value.ndim == ref.ndim == 2
                and loaded_value.shape[1] == ref.shape[1]
                and ref.shape[0] * 2 == loaded_value.shape[0] * 3
            ):
                anchor_slice = np.zeros(
                    (ref.shape[0] - loaded_value.shape[0], ref.shape[1]),
                    dtype=ref.dtype,
                )
                loaded_value = np.concatenate(
                    [loaded_value, anchor_slice], axis=0
                )
            result[k] = loaded_value.astype(ref.dtype)

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    # Unflatten back to nested dict, converting numeric key parts back to integers
    unflat = {}
    for k, v in result.items():
        parts = k.split("/")
        parsed_parts = [int(p) if p.isdigit() else p for p in parts]
        unflat[tuple(parsed_parts)] = v

    return flax.traverse_util.unflatten_dict(unflat)
