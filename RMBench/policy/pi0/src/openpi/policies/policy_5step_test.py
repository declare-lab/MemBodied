from openpi.policies import policy as base_policy
from openpi.policies import policy_5step


def test_memory_slots_advance_round_robin(monkeypatch):
    instance = policy_5step.Policy.__new__(policy_5step.Policy)
    instance._reset_slots()

    seen = []

    def fake_infer(self, obs):
        seen.append(self._memory_state)
        self._memory_state = obs["step"]
        self._prev_h_states = f"hidden-{obs['step']}"
        self._prev_actions = f"action-{obs['step']}"
        return {"actions": [obs["step"]]}

    monkeypatch.setattr(base_policy.Policy, "infer", fake_infer)
    for step in range(11):
        instance.infer({"step": step})

    assert seen[:10] == [None] * 10
    assert seen[10] == 0
    assert instance._slot_counter == 11
