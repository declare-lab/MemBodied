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


def test_adapter_interface(monkeypatch):
    stub = StubPolicy()
    monkeypatch.setattr(pi_model.training_config, "get_config", lambda name: name)
    monkeypatch.setattr(
        pi_model.policy_config,
        "create_trained_policy",
        lambda config, checkpoint, asset_id=None: stub,
    )
    adapter = pi_model.MemBodiedPolicyAdapter(
        backend="pi05",
        config_name="membodied_pi05",
        checkpoint_dir="checkpoint",
        action_chunk_size=2,
    )
    adapter.set_language("instruction")
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    adapter.update_observation_window([image, image + 1, image + 2], np.arange(14))

    assert adapter.get_action().shape == (2, 14)
    assert stub.observation["images"]["cam_high"].shape == (3, 4, 5)
    adapter.reset()
    assert stub.reset_count == 1

    with pytest.raises(ValueError, match="serves the pi05 backend"):
        pi_model.MemBodiedPolicyAdapter(
            backend="pi0",
            config_name="membodied_pi05",
            checkpoint_dir="checkpoint",
        )
