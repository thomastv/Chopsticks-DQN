"""Exact Retrograde Minimax Solver (Oracle) for Chopsticks."""

from __future__ import annotations

from collections import deque

import numpy as np

from chopsticks.env import (
    NUM_ACTIONS,
    State,
    apply_action,
    get_legal_action_mask,
    is_terminal_loss,
)


class RetrogradeOracle:
    """Exact game-theoretic solver over all 5^4 = 625 Chopsticks states.

    Uses retrograde minimax analysis for finite cyclic two-player zero-sum games:
    - Terminal loss states (own_l == 0 and own_r == 0) have value -1 at ply 0.
    - A state is a Win (+1) if at least one legal action transitions to an opponent Loss (-1).
    - A state is a Loss (-1) if all legal actions transition to opponent Wins (+1).
    - Any state remaining unresolved after fixed-point convergence is an exact Draw (0)
      under optimal play (forced cycle).
    """

    def __init__(self) -> None:
        self.all_states: list[State] = [
            State(ol, or_, pl, pr)
            for ol in range(5)
            for or_ in range(5)
            for pl in range(5)
            for pr in range(5)
        ]
        # Valid non-terminal states where neither player is already eliminated
        self.non_terminal_states: list[State] = [
            s
            for s in self.all_states
            if not (s.own_l == 0 and s.own_r == 0) and not (s.opp_l == 0 and s.opp_r == 0)
        ]
        self.values: dict[State, int] = {}
        self.plies: dict[State, int] = {}
        self.transitions: dict[State, dict[int, State]] = {}
        self.reachable_states: set[State] = set()

        self._build_graph_and_solve()
        self._compute_reachable_from_start()

    def _build_graph_and_solve(self) -> None:
        for s in self.all_states:
            if is_terminal_loss(s):
                # Active player has (0, 0) -> immediate loss
                self.values[s] = -1
                self.plies[s] = 0
                self.transitions[s] = {}
            elif s.opp_l == 0 and s.opp_r == 0:
                # Opponent already has (0, 0) (unreachable in normal play since previous step ends)
                self.values[s] = 1
                self.plies[s] = 0
                self.transitions[s] = {}
            else:
                self.values[s] = 0
                self.plies[s] = 0
                mask = get_legal_action_mask(s)
                self.transitions[s] = {
                    a: apply_action(s, a) for a in range(NUM_ACTIONS) if mask[a]
                }

        # Iteratively resolve Win (+1) and Loss (-1) states by ply depth
        changed = True
        current_ply = 0
        while changed:
            changed = False
            current_ply += 1
            for s in self.non_terminal_states:
                if self.values[s] != 0:
                    continue
                next_states = self.transitions[s]
                if not next_states:
                    continue

                # Can the active player move to an already-known opponent Loss (-1)?
                losing_children = [
                    ns for ns in next_states.values() if self.values[ns] == -1
                ]
                if losing_children:
                    self.values[s] = 1
                    self.plies[s] = min(self.plies[ns] for ns in losing_children) + 1
                    changed = True
                    continue

                # Do ALL legal actions lead to already-known opponent Wins (+1)?
                if all(self.values[ns] == 1 for ns in next_states.values()):
                    self.values[s] = -1
                    self.plies[s] = max(self.plies[ns] for ns in next_states.values()) + 1
                    changed = True

    def _compute_reachable_from_start(self, start: State = State(1, 1, 1, 1)) -> None:
        visited: set[State] = {start}
        queue: deque[State] = deque([start])
        while queue:
            curr = queue.popleft()
            for ns in self.transitions.get(curr, {}).values():
                if not is_terminal_loss(ns) and ns not in visited:
                    visited.add(ns)
                    queue.append(ns)
        self.reachable_states = visited

    def action_outcome(self, state: State, action: int) -> tuple[int, int]:
        """Return (mover_outcome, plies_to_terminal) for taking `action` in `state`.

        mover_outcome is in {+1 (Win), 0 (Draw), -1 (Loss)}.
        """
        next_state = self.transitions[state][action]
        outcome = -self.values[next_state]
        ply = self.plies[next_state] + 1 if outcome != 0 else 0
        return outcome, ply

    def exact_discounted_q(self, state: State, action: int, gamma: float = 0.99) -> float:
        """Return the exact game-theoretic discounted Q*(s, a) value."""
        outcome, ply = self.action_outcome(state, action)
        if outcome == 0:
            return 0.0
        # Reward of +/-1 is received after `ply` transitions (discount factor gamma^(ply - 1))
        return float(outcome) * (gamma ** (ply - 1))

    def optimal_value_actions(self, state: State) -> list[int]:
        """Return all legal actions that achieve the optimal game-theoretic outcome (+1/0/-1)."""
        legal_actions = list(self.transitions[state].keys())
        if not legal_actions:
            return []
        outcomes = {a: self.action_outcome(state, a)[0] for a in legal_actions}
        best_outcome = max(outcomes.values())
        return [a for a in legal_actions if outcomes[a] == best_outcome]

    def best_actions_with_ply(self, state: State) -> list[int]:
        """Return legal actions that achieve the optimal outcome AND fastest win / slowest loss."""
        legal_actions = list(self.transitions[state].keys())
        if not legal_actions:
            return []

        def sort_key(a: int) -> tuple[int, int]:
            outcome, ply = self.action_outcome(state, a)
            if outcome == 1:
                # Win: prefer higher outcome (1), then smaller ply (-ply)
                return (1, -ply)
            if outcome == 0:
                return (0, 0)
            # Loss: prefer higher ply (delay loss as long as possible)
            return (-1, ply)

        best_key = max(sort_key(a) for a in legal_actions)
        return [a for a in legal_actions if sort_key(a) == best_key]

    def select_action(self, state: State, rng: np.random.Generator | None = None) -> int:
        """Select a game-theoretically optimal move (fastest win / draw / slowest loss)."""
        candidates = self.best_actions_with_ply(state)
        if not candidates:
            raise ValueError(f"No legal actions available in state {state}")
        if rng is not None:
            return int(rng.choice(candidates))
        return candidates[0]

    def summary_stats(self) -> dict[str, int]:
        """Return counts of Win, Draw, and Loss states across non-terminal and reachable sets."""
        non_term_counts = {
            "total_non_terminal": len(self.non_terminal_states),
            "win": sum(1 for s in self.non_terminal_states if self.values[s] == 1),
            "draw": sum(1 for s in self.non_terminal_states if self.values[s] == 0),
            "loss": sum(1 for s in self.non_terminal_states if self.values[s] == -1),
            "reachable_from_start": len(self.reachable_states),
            "reachable_win": sum(1 for s in self.reachable_states if self.values[s] == 1),
            "reachable_draw": sum(1 for s in self.reachable_states if self.values[s] == 0),
            "reachable_loss": sum(1 for s in self.reachable_states if self.values[s] == -1),
            "start_state_value": self.values[State(1, 1, 1, 1)],
            "start_state_ply": self.plies[State(1, 1, 1, 1)],
        }
        return non_term_counts
