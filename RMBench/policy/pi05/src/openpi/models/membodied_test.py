import pytest
import jax.numpy as jnp
import numpy as np

from openpi.models.memory_config import migrate_legacy_config
from openpi.models.pi05_mem_vector import Pi0MemConfig
from openpi.models.pi05_mem_vector import anchor_update
from openpi.models.pi05_mem_vector import resolve_first_frame_sequence
from openpi.training import config as training_config


@pytest.mark.parametrize("memory_type", ["base", "lstm_cell"])
def test_public_memory_types(memory_type):
    assert Pi0MemConfig(memory_type=memory_type).memory_type == memory_type


@pytest.mark.parametrize("memory_type", ["decoupled", "linear_unroll", "cross_state_agg", "cross_state_attn"])
def test_removed_memory_types_are_rejected(memory_type):
    with pytest.raises(ValueError, match="Unsupported memory_type"):
        Pi0MemConfig(memory_type=memory_type)


def test_legacy_metadata_migration():
    assert migrate_legacy_config({
        "memory_type": "decoupled",
        "memory_rank": 128,
        "memory_key_source": "state",
        "anchor_site": "prefix",
    }) == {"memory_type": "base", "memory_rank": 128}


def test_anchor_and_first_frame_reset_semantics():
    previous = jnp.zeros((2, 4, 3))
    current = jnp.ones((2, 4, 3))
    result = anchor_update(previous, current, jnp.array([True, False]))
    np.testing.assert_array_equal(result[0], current[0])
    np.testing.assert_array_equal(result[1], previous[1])

    candidates = jnp.arange(5, dtype=jnp.float32).reshape(1, 5, 1)
    frames = resolve_first_frame_sequence(
        candidates, jnp.array([[True, False, False, True, False]])
    )
    np.testing.assert_array_equal(frames[..., 0], [[0, 0, 0, 3, 3]])


def test_canonical_pi05_config():
    assert [config.name for config in training_config._CONFIGS] == ["membodied_pi05"]
    model = training_config.get_config("membodied_pi05").model
    assert model.memory_type == "base"
    assert model.memory_value_type == "vision_action"
    assert (model.sequence_len, model.memory_rank, model.memory_alpha) == (8, 128, 256)
    assert model.use_anchor_memory is True

