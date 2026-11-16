"""Core Chopsticks game environment (Rollover mod 5 + Division-only split)."""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

import numpy as np

NUM_ACTIONS = 5
STATE_DIM = 20  # 4 hands * 5 possible finger counts (0..4)

ACTION_NAMES: dict[int, str] = {
    0: "Attack L -> L",
    1: "Attack L -> R",
    2: "Attack R -> L",
    3: "Attack R -> R",
    4: "Split (Divide)",
}


class State(NamedTuple):
    """Turn-relative state from the active player's perspective."""

    own_l: int
    own_r: int
    opp_l: int
    opp_r: int

    def canonical(self) -> State:
        """Return left/right order-invariant canonical state."""
        ol, or_ = (self.own_l, self.own_r) if self.own_l <= self.own_r else (self.own_r, self.own_l)
        pl, pr = (self.opp_l, self.opp_r) if self.opp_l <= self.opp_r else (self.opp_r, self.opp_l)
        return State(ol, or_, pl, pr)


def is_terminal_loss(state: State) -> bool:
    """Return True if the active player has both hands dead (0) and has lost."""
    return state.own_l == 0 and state.own_r == 0


def get_legal_action_mask(state: State) -> np.ndarray:
    """Return a boolean mask of shape (5,) indicating legal actions in `state`."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    if is_terminal_loss(state) or (state.opp_l == 0 and state.opp_r == 0):
        return mask

    # 0: own_l -> opp_l
    mask[0] = state.own_l > 0 and state.opp_l > 0
    # 1: own_l -> opp_r
    mask[1] = state.own_l > 0 and state.opp_r > 0
    # 2: own_r -> opp_l
    mask[2] = state.own_r > 0 and state.opp_l > 0
    # 3: own_r -> opp_r
    mask[3] = state.own_r > 0 and state.opp_r > 0
    # 4: Division-only split when one hand is dead (0) and the other is even (2 or 4)
    mask[4] = (state.own_l == 0 and state.own_r in (2, 4)) or (
        state.own_r == 0 and state.own_l in (2, 4)
    )
    return mask


def apply_action(state: State, action: int) -> State:
    """Apply `action` in `state` and return the next player's turn-relative `State`."""
    if not (0 <= action < NUM_ACTIONS):
        raise ValueError(f"Action must be in 0..{NUM_ACTIONS - 1}, got {action}")
    mask = get_legal_action_mask(state)
    if not mask[action]:
        raise ValueError(f"Illegal action {action} ({ACTION_NAMES.get(action)}) in state {state}")

    own_l, own_r, opp_l, opp_r = state

    if action == 0:
        opp_l = (opp_l + own_l) % 5
    elif action == 1:
        opp_r = (opp_r + own_l) % 5
    elif action == 2:
        opp_l = (opp_l + own_r) % 5
    elif action == 3:
        opp_r = (opp_r + own_r) % 5
    elif action == 4:
        total = own_l + own_r
        half = total // 2
        own_l, own_r = half, half

    # Flip perspective for the next player's turn
    return State(opp_l, opp_r, own_l, own_r)


def encode_state(state: State) -> np.ndarray:
    """Encode `State` as a 20-dimensional one-hot float32 vector."""
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    vec[0 + state.own_l] = 1.0
    vec[5 + state.own_r] = 1.0
    vec[10 + state.opp_l] = 1.0
    vec[15 + state.opp_r] = 1.0
    return vec


class ChopsticksEnv:
    """Two-player alternating zero-sum Chopsticks environment."""

    def __init__(self, max_steps: int = 100, repetition_limit: int = 3) -> None:
        self.max_steps = max_steps
        self.repetition_limit = repetition_limit
        self.state: State = State(1, 1, 1, 1)
        self.current_player: int = 0
        self.steps: int = 0
        self.done: bool = False
        self.winner: int | None = None
        self.termination_reason: str | None = None
        self._history_counts: Counter[tuple[int, State]] = Counter()

    def reset(
        self,
        initial_state: State | None = None,
        starting_player: int = 0,
    ) -> tuple[State, np.ndarray]:
        """Reset environment to initial state (default (1, 1, 1, 1)) and return (state, legal_mask)."""
        self.state = initial_state if initial_state is not None else State(1, 1, 1, 1)
        self.current_player = starting_player
        self.steps = 0
        self.done = False
        self.winner = None
        self.termination_reason = None
        self._history_counts.clear()
        self._record_current_position()
        return self.state, get_legal_action_mask(self.state)

    def _record_current_position(self) -> int:
        key = (self.current_player, self.state.canonical())
        self._history_counts[key] += 1
        return self._history_counts[key]

    def get_player_hands(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return ((p0_l, p0_r), (p1_l, p1_r)) in absolute player order."""
        if self.current_player == 0:
            return (self.state.own_l, self.state.own_r), (self.state.opp_l, self.state.opp_r)
        return (self.state.opp_l, self.state.opp_r), (self.state.own_l, self.state.own_r)

    def step(self, action: int) -> tuple[State, float, bool, dict[str, object]]:
        """Execute `action` for `current_player`.

        Returns:
            next_state: Turn-relative state from the NEXT player's perspective.
            reward: Reward for the player who just acted (+1.0 win, 0.0 draw/ongoing).
            done: Whether the episode has terminated.
            info: Metadata dictionary (winner, reason, next_legal_mask).
        """
        if self.done:
            raise RuntimeError("Cannot call step() on a terminated episode. Call reset() first.")

        acting_player = self.current_player
        next_state = apply_action(self.state, action)
        self.state = next_state
        self.current_player = 1 - acting_player
        self.steps += 1

        # Check if the next player has no live hands -> acting_player won!
        if is_terminal_loss(next_state):
            self.done = True
            self.winner = acting_player
            self.termination_reason = "elimination"
            return (
                next_state,
                1.0,
                True,
                {
                    "winner": self.winner,
                    "reason": self.termination_reason,
                    "legal_mask": get_legal_action_mask(next_state),
                },
            )

        # Record new position and check 3-fold state repetition
        rep_count = self._record_current_position()
        if self.repetition_limit > 0 and rep_count >= self.repetition_limit:
            self.done = True
            self.winner = None
            self.termination_reason = "repetition"
            return (
                next_state,
                0.0,
                True,
                {
                    "winner": None,
                    "reason": self.termination_reason,
                    "legal_mask": get_legal_action_mask(next_state),
                },
            )

        # Check step limit draw
        if self.steps >= self.max_steps:
            self.done = True
            self.winner = None
            self.termination_reason = "max_steps"
            return (
                next_state,
                0.0,
                True,
                {
                    "winner": None,
                    "reason": self.termination_reason,
                    "legal_mask": get_legal_action_mask(next_state),
                },
            )

        return (
            next_state,
            0.0,
            False,
            {
                "winner": None,
                "reason": None,
                "legal_mask": get_legal_action_mask(next_state),
            },
        )
