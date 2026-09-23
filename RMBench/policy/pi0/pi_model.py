"""Public RMBench adapter for the MemBodied pi0 backend."""

import os
import pathlib

import numpy as np

from openpi.policies import policy_config
from openpi.training import config as training_config


class MemBodiedPolicyAdapter:
    def __init__(
        self,
        *,
        backend: str,
        config_name: str,
        checkpoint_dir: str | os.PathLike[str] | None,
        asset_id: str | None = None,
        action_chunk_size: int = 50,
    ):
        if backend != "pi0":
            raise ValueError(
                f"This environment serves the pi0 backend, not {backend!r}. "
                "Activate the selected backend environment before evaluation."
            )
        if action_chunk_size < 1:
            raise ValueError("action_chunk_size must be positive")

        configured_checkpoint = checkpoint_dir or os.getenv("MEMBODIED_CHECKPOINT_DIR")
        if not configured_checkpoint:
            raise ValueError(
                "checkpoint_dir is required; set it in YAML/CLI or define "
                "MEMBODIED_CHECKPOINT_DIR"
            )

        self.checkpoint_dir = pathlib.Path(configured_checkpoint).expanduser()
        self.config = training_config.get_config(config_name)
        self.policy = policy_config.create_trained_policy(
            self.config,
            self.checkpoint_dir,
            asset_id=asset_id,
        )
        self.action_chunk_size = action_chunk_size
        self.observation_window = None
        self.instruction = None

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def update_observation_window(self, images, state) -> None:
        front, right, left = images
        self.observation_window = {
            "state": np.asarray(state),
            "images": {
                "cam_high": np.transpose(front, (2, 0, 1)),
                "cam_left_wrist": np.transpose(left, (2, 0, 1)),
                "cam_right_wrist": np.transpose(right, (2, 0, 1)),
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        if self.observation_window is None:
            raise RuntimeError("update_observation_window must be called before get_action")
        actions = self.policy.infer(self.observation_window)["actions"]
        return actions[: self.action_chunk_size]

    def reset(self) -> None:
        self.instruction = None
        self.observation_window = None
        self.policy.reset()


PI0 = MemBodiedPolicyAdapter

