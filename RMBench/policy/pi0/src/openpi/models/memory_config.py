"""Compatibility helpers for pre-release MemBodied checkpoint metadata."""

from collections.abc import Mapping
from typing import Any


_OBSOLETE_FIELDS = frozenset({
    "memory_pos_dim",
    "memory_feature_map",
    "memory_pooling_type",
    "memory_action_pooling",
    "memory_unified_attention",
    "memory_key_activation",
    "gate_reg_coeff",
    "anchor_top_camera_only",
    "anchor_mode",
    "anchor_grid",
    "anchor_rank",
    "anchor_site",
    "use_delta_memory",
    "freeze_lora",
    "memory_key_source",
})


def migrate_legacy_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return metadata accepted by the public MemBodied configs.

    This intentionally supports only the checkpoint metadata emitted by the
    private development repository. Removed experiment modes remain invalid.
    """
    migrated = dict(metadata)
    if migrated.get("memory_type") == "decoupled":
        migrated["memory_type"] = "base"
    for field in _OBSOLETE_FIELDS:
        migrated.pop(field, None)
    return migrated

