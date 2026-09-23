import pytest
import jax.numpy as jnp
import numpy as np

from openpi.models.memory_config import migrate_legacy_config
from openpi.models.pi0_mem import Pi0MemConfig as AttentionConfig
from openpi.models.pi0_mem_vector import Pi0MemConfig as VectorConfig
from openpi.models.pi0_mem_vector import anchor_update
from openpi.models.pi0_mem_vector import resolve_first_frame_sequence
from openpi.training import config as training_config


@pytest.mark.parametrize("config_cls", [AttentionConfig, VectorConfig])
@pytest.mark.parametrize("memory_type", ["base", "lstm_cell"])
def test_public_memory_types(config_cls, memory_type):
    config = config_cls(memory_type=memory_type)
    assert config.memory_type == memory_type


@pytest.mark.parametrize("config_cls", [AttentionConfig, VectorConfig])
@pytest.mark.parametrize("memory_type", ["decoupled", "linear_unroll", "cross_state_agg", "cross_state_attn"])
def test_removed_memory_types_are_rejected(config_cls, memory_type):
    with pytest.raises(ValueError, match="Unsupported memory_type"):
        config_cls(memory_type=memory_type)


@pytest.mark.parametrize("value_type", ["vision_action", "vision", "action"])
def test_public_value_compositions(value_type):
    assert VectorConfig(memory_value_type=value_type).memory_value_type == value_type


def test_legacy_metadata_migration_is_narrow():
    migrated = migrate_legacy_config({
        "memory_type": "decoupled",
        "memory_rank": 128,
        "anchor_grid": 4,
        "use_delta_memory": True,
    })
    assert migrated == {"memory_type": "base", "memory_rank": 128}


def test_anchor_update_resets_only_new_episodes():
    previous = jnp.zeros((2, 4, 3))
    current = jnp.ones((2, 4, 3))
    result = anchor_update(previous, current, jnp.array([True, False]))
    np.testing.assert_array_equal(result[0], current[0])
    np.testing.assert_array_equal(result[1], previous[1])


def test_first_frame_sequence_switches_at_episode_boundaries():
    candidates = jnp.arange(5, dtype=jnp.float32).reshape(1, 5, 1)
    is_first = jnp.array([[True, False, False, True, False]])
    result = resolve_first_frame_sequence(candidates, is_first)
    np.testing.assert_array_equal(result[..., 0], [[0, 0, 0, 3, 3]])


def test_canonical_configs_match_paper_variants():
    configs = {config.name: config for config in training_config._CONFIGS}
    assert list(configs) == [
        "membodied",
        "membodied_no_anchor",
        "membodied_as",
        "membodied_h",
        "membodied_vision_only",
        "membodied_action_only",
        "membodied_first_frame",
        "membodied_libero",
    ]
    assert configs["membodied"].model.use_anchor_memory is True
    assert configs["membodied_no_anchor"].model.use_anchor_memory is False
    assert configs["membodied_h"].model.memory_type == "lstm_cell"
    assert configs["membodied_vision_only"].model.memory_value_type == "vision"
    assert configs["membodied_action_only"].model.memory_value_type == "action"
    assert configs["membodied_first_frame"].model.use_first_frame is True
    libero = configs["membodied_libero"].model
    assert (libero.sequence_len, libero.memory_rank, libero.memory_alpha) == (6, 8, 16)
    assert libero.memory_dropout == 0.2
    assert libero.mem_init is False

