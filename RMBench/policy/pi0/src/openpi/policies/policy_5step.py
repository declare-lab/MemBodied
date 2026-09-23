"""Five-step replanning policy for LIBERO.

Ten independent memory streams are interleaved so each stream advances once per
50 environment steps while the controller replans every five steps.
"""

from openpi.policies.policy import Policy as _Policy
from openpi.policies.policy import PolicyRecorder


class Policy(_Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_slots()

    def _reset_slots(self) -> None:
        self._memory_states = [None] * 10
        self._previous_hidden_states = [None] * 10
        self._previous_actions = [None] * 10
        self._slot_counter = 0

    def reset(self) -> None:
        super().reset()
        self._reset_slots()

    def infer(self, obs: dict) -> dict:
        slot = self._slot_counter % 10
        self._memory_state = self._memory_states[slot]
        self._prev_h_states = self._previous_hidden_states[slot]
        self._prev_actions = self._previous_actions[slot]

        result = super().infer(obs)

        self._memory_states[slot] = self._memory_state
        self._previous_hidden_states[slot] = self._prev_h_states
        self._previous_actions[slot] = self._prev_actions
        self._slot_counter += 1
        return result


__all__ = ["Policy", "PolicyRecorder"]

