import numpy as np
import pytest

import pi_model


class StubPolicy:
    def __init__(self):
        self.observation = None
        self.reset_count = 0

    def infer(self, observation):
        self.observation = observation
        return {"actions": np.arange(42).reshape(3, 14)}

    def reset(self):
        self.reset_count += 1


def _adapter(monkeypatch, **kwargs):
    stub = StubPolicy()
    monkeypatch.setattr(pi_model.training_config, "get_config", lambda name: name)
    monkeypatch.setattr(
        pi_model.policy_config,
        "create_trained_policy",
        lambda config, checkpoint, asset_id=None: stub,
    )
    adapter = pi_model.MemBodiedPolicyAdapter(
        backend="pi0",
        config_name="membodied",
        checkpoint_dir="checkpoint",
        action_chunk_size=2,
        **kwargs,
    )
    return adapter, stub


def test_adapter_encodes_observation_chunks_actions_and_resets(monkeypatch):
    adapter, stub = _adapter(monkeypatch)
    adapter.set_language("instruction")
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    adapter.update_observation_window([image, image + 1, image + 2], np.arange(14))

    assert adapter.get_action().shape == (2, 14)
    assert stub.observation["images"]["cam_high"].shape == (3, 4, 5)
    assert stub.observation["prompt"] == "instruction"

    adapter.reset()
    assert adapter.observation_window is None
    assert adapter.instruction is None
    assert stub.reset_count == 1


def test_checkpoint_environment_fallback_and_errors(monkeypatch):
    monkeypatch.setenv("MEMBODIED_CHECKPOINT_DIR", "from-env")
    monkeypatch.setattr(pi_model.training_config, "get_config", lambda name: name)
    monkeypatch.setattr(
        pi_model.policy_config,
        "create_trained_policy",
        lambda *args, **kwargs: StubPolicy(),
    )
    adapter = pi_model.MemBodiedPolicyAdapter(
        backend="pi0", config_name="membodied", checkpoint_dir=None
    )
    assert adapter.checkpoint_dir.name == "from-env"

    with pytest.raises(ValueError, match="serves the pi0 backend"):
        pi_model.MemBodiedPolicyAdapter(
            backend="pi05", config_name="membodied", checkpoint_dir="checkpoint"
        )
