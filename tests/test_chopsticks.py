"""Unit tests for Chopsticks environment, Retrograde Oracle, and Double Dueling DQN."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chopsticks.dqn import DQNAgent, DuelingDQN
from chopsticks.env import (
    NUM_ACTIONS,
    ChopsticksEnv,
    State,
    apply_action,
    encode_state,
    get_legal_action_mask,
    is_terminal_loss,
)
from chopsticks.oracle import RetrogradeOracle


def test_rollover_mod5_arithmetic() -> None:
    # 3 + 4 = 7 -> 2 (mod 5), next player's perspective is (opp_l, opp_r, own_l, own_r)
    state = State(3, 1, 4, 2)
    next_state = apply_action(state, 0)  # own_l (3) attacks opp_l (4) -> 2
    assert next_state == State(2, 2, 3, 1)

    # Exact 5 dies: 3 + 2 = 5 -> 0
    next_state_kill = apply_action(state, 1)  # own_l (3) attacks opp_r (2) -> 0
    assert next_state_kill == State(4, 0, 3, 1)


def test_division_only_split_legality() -> None:
    # Even live hand + dead hand -> split is legal
    for live in (2, 4):
        s_left = State(live, 0, 1, 1)
        s_right = State(0, live, 1, 1)
        assert get_legal_action_mask(s_left)[4]
        assert get_legal_action_mask(s_right)[4]
        assert apply_action(s_left, 4) == State(1, 1, live // 2, live // 2)
        assert apply_action(s_right, 4) == State(1, 1, live // 2, live // 2)

    # Odd live hand + dead hand -> split is illegal
    for odd in (1, 3):
        assert not get_legal_action_mask(State(odd, 0, 1, 1))[4]
        assert not get_legal_action_mask(State(0, odd, 1, 1))[4]

    # Both hands alive -> split is illegal even if sum is even
    for s in [State(1, 1, 1, 1), State(2, 2, 1, 1), State(3, 1, 1, 1), State(4, 2, 1, 1)]:
        assert not get_legal_action_mask(s)[4]
        with pytest.raises(ValueError, match="Illegal action"):
            apply_action(s, 4)


def test_terminal_elimination_in_env() -> None:
    env = ChopsticksEnv()
    # Player 0 has (3, 0), Player 1 has (0, 2) -> Player 0 attacks L->R (action 1) and wins
    env.reset(initial_state=State(3, 0, 0, 2), starting_player=0)
    next_state, reward, done, info = env.step(1)
    assert is_terminal_loss(next_state)
    assert done is True
    assert reward == 1.0
    assert env.winner == 0
    assert info["reason"] == "elimination"


def test_threefold_repetition_draw() -> None:
    env = ChopsticksEnv(max_steps=100, repetition_limit=3)
    # Construct a cyclic sequence using splits:
    # P0: (2, 0), P1: (2, 0)
    # Step 1: P0 splits -> (1, 1); Step 2: P1 attacks L->L (2+1=3) etc.
    # Or directly test repetition of canonical state via split/attack loop:
    # Suppose P0 has (4, 0) and P1 has (3, 0):
    # Wait: let's verify a 4-ply cycle:
    # State(2, 0, 2, 0) is not directly cyclic by split alone, let's check an actual cycle:
    # P0=(1, 0), P1=(2, 0):
    # 1. P0 L->L: P1 becomes (3, 0) [State(3, 0, 1, 0)]
    # 2. P1 L->L: P0 becomes (4, 0) [State(4, 0, 3, 0)]
    # 3. P0 L->L: P1 becomes (2, 0) [State(2, 0, 4, 0)]
    # 4. P1 L->L: P0 becomes (1, 0) [State(1, 0, 2, 0)] -> back to initial state!
    state, _ = env.reset(initial_state=State(1, 0, 2, 0), starting_player=0)
    assert state == State(1, 0, 2, 0)

    # First full cycle (4 steps) brings us to 2nd occurrence of (P0, State(1, 0, 2, 0))
    for _ in range(4):
        state, reward, done, _ = env.step(0)
    assert state == State(1, 0, 2, 0)
    assert not done
    assert reward == 0.0

    # Second full cycle (4 steps) hits the 3rd occurrence -> Draw by repetition!
    for i in range(4):
        state, reward, done, info = env.step(0)
        if i < 3:
            assert not done
        else:
            assert done
            assert reward == 0.0
            assert env.winner is None
            assert info["reason"] == "repetition"


def test_retrograde_oracle_consistency() -> None:
    oracle = RetrogradeOracle()
    assert len(oracle.non_terminal_states) == 576

    for s in oracle.non_terminal_states:
        val = oracle.values[s]
        children = list(oracle.transitions[s].values())
        assert len(children) > 0

        if val == 1:
            # Must have at least one move to an opponent Loss (-1)
            losing_children = [c for c in children if oracle.values[c] == -1]
            assert len(losing_children) >= 1
            assert oracle.plies[s] == min(oracle.plies[c] for c in losing_children) + 1
        elif val == -1:
            # All moves must lead to opponent Wins (+1)
            assert all(oracle.values[c] == 1 for c in children)
            assert oracle.plies[s] == max(oracle.plies[c] for c in children) + 1
        else:
            # Draw (0): cannot have any move to -1, and must have at least one move to 0
            assert all(oracle.values[c] != -1 for c in children)
            assert any(oracle.values[c] == 0 for c in children)


def test_dueling_dqn_masking_and_bellman_step() -> None:
    net = DuelingDQN(hidden_dim=64)
    s = State(2, 0, 3, 0)
    mask = get_legal_action_mask(s)
    # Legal actions for (2, 0, 3, 0): 0 (L->L) and 4 (Split)
    assert list(np.flatnonzero(mask)) == [0, 4]

    s_vec = torch.from_numpy(encode_state(s))
    mask_t = torch.from_numpy(mask)
    q = net(s_vec, mask_t)
    assert q.shape == (NUM_ACTIONS,)

    agent = DQNAgent(lr=1e-2, gamma=0.95, tau=0.05, hidden_dim=64, seed=7)
    q_masked = agent.get_q_values(s, mask_illegal=True)
    assert np.isfinite(q_masked[0]) and np.isfinite(q_masked[4])
    assert np.isneginf(q_masked[1]) and np.isneginf(q_masked[2]) and np.isneginf(q_masked[3])

    # Verify winning transition converges toward +1.0
    win_state = State(3, 0, 2, 0)  # Action 0 kills opp_l (3+2=5->0) -> terminal win (+1.0)
    win_mask = get_legal_action_mask(win_state)
    next_s = apply_action(win_state, 0)
    next_mask = get_legal_action_mask(next_s)

    for _ in range(64):
        agent.buffer.push(
            state_vec=encode_state(win_state),
            legal_mask=win_mask,
            action=0,
            reward=1.0,
            next_state_vec=encode_state(next_s),
            next_legal_mask=next_mask,
            done=True,
        )
    for _ in range(50):
        loss = agent.update(batch_size=32)
        assert loss is not None

    learned_q = agent.get_q_values(win_state, mask_illegal=True)[0]
    assert learned_q > 0.90
