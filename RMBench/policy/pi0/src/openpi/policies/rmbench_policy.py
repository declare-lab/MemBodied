import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class RMBenchInputs(transforms.DataTransformFn):
    """Convert RMBench observations to OpenPI's camera/state schema."""

    action_dim: int
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(np.asarray(data["state"]), self.action_dim)
        source_images = data["images"]
        unknown = set(source_images) - set(self.EXPECTED_CAMERAS)
        if unknown:
            raise ValueError(f"Unexpected RMBench cameras: {sorted(unknown)}")

        def image(name: str) -> np.ndarray:
            value = np.asarray(source_images[name])
            if np.issubdtype(value.dtype, np.floating):
                value = (255 * value).astype(np.uint8)
            return einops.rearrange(value, "c h w -> h w c")

        base = image("cam_high")
        images = {"base_0_rgb": base}
        masks = {"base_0_rgb": np.True_}
        for target, source in (
            ("left_wrist_0_rgb", "cam_left_wrist"),
            ("right_wrist_0_rgb", "cam_right_wrist"),
        ):
            if source in source_images:
                images[target] = image(source)
                masks[target] = np.True_
            else:
                images[target] = np.zeros_like(base)
                masks[target] = np.False_

        result = {"image": images, "image_mask": masks, "state": state}
        if "actions" in data:
            result["actions"] = transforms.pad_to_dim(
                np.asarray(data["actions"]), self.action_dim
            )
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class RMBenchOutputs(transforms.DataTransformFn):
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}

