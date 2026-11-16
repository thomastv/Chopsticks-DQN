"""Self-play training and exact oracle evaluation for the Chopsticks DQN agent."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from chopsticks.dqn import DQNAgent
from chopsticks.env import (
    NUM_ACTIONS,
    ChopsticksEnv,
    State,
    encode_state,
    get_legal_action_mask,
    is_terminal_loss,
)
from chopsticks.oracle import RetrogradeOracle


def evaluate_against_oracle(
    agent: DQNAgent,
    oracle: RetrogradeOracle,
) -> dict[str, float]:
    """Compare the DQN's greedy policy and Q-values against the Exact Retrograde Oracle."""
    reachable_value_correct = 0
    reachable_ply_correct = 0
    all_value_correct = 0
    all_ply_correct = 0
    q_abs_errors: list[float] = []

    for state in oracle.non_terminal_states:
        q_vals = agent.get_q_values(state, mask_illegal=True)
        greedy_action = int(np.argmax(q_vals))

        opt_val_actions = oracle.optimal_value_actions(state)
        opt_ply_actions = oracle.best_actions_with_ply(state)

        is_val_opt = greedy_action in opt_val_actions
        is_ply_opt = greedy_action in opt_ply_actions

        if is_val_opt:
            all_value_correct += 1
        if is_ply_opt:
            all_ply_correct += 1

        if state in oracle.reachable_states:
            if is_val_opt:
                reachable_value_correct += 1
            if is_ply_opt:
                reachable_ply_correct += 1

        for action in range(NUM_ACTIONS):
            if np.isfinite(q_vals[action]):
                exact_q = oracle.exact_discounted_q(state, action, gamma=agent.gamma)
                q_abs_errors.append(abs(float(q_vals[action]) - exact_q))

    n_all = len(oracle.non_terminal_states)
    n_reach = len(oracle.reachable_states)

    return {
        "reachable_value_acc": 100.0 * reachable_value_correct / max(n_reach, 1),
        "reachable_ply_acc": 100.0 * reachable_ply_correct / max(n_reach, 1),
        "all_value_acc": 100.0 * all_value_correct / max(n_all, 1),
        "all_ply_acc": 100.0 * all_ply_correct / max(n_all, 1),
        "mean_q_mae": float(np.mean(q_abs_errors)) if q_abs_errors else 0.0,
    }


def play_match_series(
    agent: DQNAgent,
    opponent_type: str,
    oracle: RetrogradeOracle,
    num_games_per_side: int = 100,
    seed: int = 123,
) -> dict[str, dict[str, int]]:
    """Play `num_games_per_side` games as P1 and as P2 against `random` or `oracle`."""
    rng = np.random.default_rng(seed)
    env = ChopsticksEnv(max_steps=100, repetition_limit=3)
    results: dict[str, dict[str, int]] = {
        "as_p1": {"wins": 0, "draws": 0, "losses": 0},
        "as_p2": {"wins": 0, "draws": 0, "losses": 0},
    }

    for dqn_player, side_key in [(0, "as_p1"), (1, "as_p2")]:
        for _ in range(num_games_per_side):
            state, legal_mask = env.reset(starting_player=0)
            done = False
            while not done:
                if env.current_player == dqn_player:
                    action = agent.select_action(state, legal_mask, epsilon=0.0)
                else:
                    if opponent_type == "random":
                        legal_indices = np.flatnonzero(legal_mask)
                        action = int(rng.choice(legal_indices))
                    elif opponent_type == "oracle":
                        action = oracle.select_action(state, rng=rng)
                    else:
                        raise ValueError(f"Unknown opponent_type: {opponent_type}")

                state, _, done, info = env.step(action)
                legal_mask = info["legal_mask"]  # type: ignore[assignment]

            if env.winner == dqn_player:
                results[side_key]["wins"] += 1
            elif env.winner is None:
                results[side_key]["draws"] += 1
            else:
                results[side_key]["losses"] += 1

    return results


def train_dqn(
    episodes: int = 6000,
    batch_size: int = 128,
    lr: float = 1.5e-3,
    gamma: float = 0.95,
    tau: float = 0.02,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    exploring_starts_prob: float = 0.7,
    eval_every: int = 1000,
    checkpoint_path: str | Path = "checkpoints/dqn_chopsticks.pt",
    seed: int = 42,
) -> tuple[DQNAgent, RetrogradeOracle, dict[str, float]]:
    """Train the Action-Masked Double Dueling DQN via self-play with exploring starts."""
    oracle = RetrogradeOracle()
    stats = oracle.summary_stats()
    print("=" * 72)
    print("Chopsticks Exact Retrograde Oracle Summary:")
    print(
        f"  Non-terminal states : {stats['total_non_terminal']} "
        f"(Win: {stats['win']}, Draw: {stats['draw']}, Loss: {stats['loss']})"
    )
    print(
        f"  Reachable from start: {stats['reachable_from_start']} "
        f"(Win: {stats['reachable_win']}, Draw: {stats['reachable_draw']}, Loss: {stats['reachable_loss']})"
    )
    verdict_map = {1: "First-Player Win (+1)", 0: "Draw (0)", -1: "Second-Player Win (-1)"}
    print(
        f"  Start state (1,1,1,1): {verdict_map[stats['start_state_value']]} "
        f"(ply depth: {stats['start_state_ply']})"
    )
    print("=" * 72)

    agent = DQNAgent(
        lr=lr,
        gamma=gamma,
        tau=tau,
        hidden_dim=128,
        buffer_capacity=50_000,
        seed=seed,
    )
    env = ChopsticksEnv(max_steps=60, repetition_limit=3)
    rng = np.random.default_rng(seed)

    non_term_list = oracle.non_terminal_states
    best_score = -1.0
    best_metrics: dict[str, float] = {}
    start_time = time.perf_counter()
    recent_losses: list[float] = []
    total_steps = 0
    train_freq = 4

    for ep in range(1, episodes + 1):
        # Linear epsilon decay over first 80% of training
        progress = min(1.0, ep / max(1, int(episodes * 0.8)))
        epsilon = eps_start + progress * (eps_end - eps_start)

        # Exploring starts ensure full coverage of all reachable & off-policy branches
        if rng.random() < exploring_starts_prob:
            init_state = non_term_list[int(rng.integers(0, len(non_term_list)))]
        else:
            init_state = State(1, 1, 1, 1)

        state, legal_mask = env.reset(initial_state=init_state)
        done = False

        while not done:
            state_vec = encode_state(state)
            action = agent.select_action(state, legal_mask, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)
            next_legal_mask: np.ndarray = info["legal_mask"]  # type: ignore[assignment]
            total_steps += 1

            # Distinguish true terminal elimination from repetition/step truncation
            terminated_by_elimination = is_terminal_loss(next_state)

            agent.buffer.push(
                state_vec=state_vec,
                legal_mask=legal_mask,
                action=action,
                reward=reward if terminated_by_elimination else 0.0,
                next_state_vec=encode_state(next_state),
                next_legal_mask=next_legal_mask,
                done=terminated_by_elimination,
            )

            if total_steps % train_freq == 0:
                loss = agent.update(batch_size=batch_size)
                if loss is not None:
                    recent_losses.append(loss)

            state = next_state
            legal_mask = next_legal_mask

        if ep % eval_every == 0 or ep == episodes:
            metrics = evaluate_against_oracle(agent, oracle)
            avg_loss = float(np.mean(recent_losses[-500:])) if recent_losses else 0.0
            elapsed = time.perf_counter() - start_time
            print(
                f"[Ep {ep:5d}/{episodes}] "
                f"eps={epsilon:.2f} | loss={avg_loss:.5f} | "
                f"Reachable Opt={metrics['reachable_value_acc']:5.1f}% (Ply={metrics['reachable_ply_acc']:5.1f}%) | "
                f"All Opt={metrics['all_value_acc']:5.1f}% | "
                f"Q-MAE={metrics['mean_q_mae']:.4f} | {elapsed:.1f}s",
                flush=True,
            )

            score = (
                metrics["reachable_value_acc"] * 10.0
                + metrics["all_value_acc"] * 5.0
                + metrics["reachable_ply_acc"]
                - metrics["mean_q_mae"] * 100.0
            )
            if score >= best_score:
                best_score = score
                best_metrics = metrics
                agent.save(checkpoint_path)

    # Load best checkpoint for final evaluation
    if Path(checkpoint_path).exists():
        agent.load(checkpoint_path)
        best_metrics = evaluate_against_oracle(agent, oracle)

    print("\n" + "=" * 72)
    print(f"Best Checkpoint Loaded from: {checkpoint_path}")
    print(
        f"  Reachable Optimal Value Accuracy : {best_metrics['reachable_value_acc']:.2f}% "
        f"(Fastest/Slowest Ply Accuracy: {best_metrics['reachable_ply_acc']:.2f}%)"
    )
    print(
        f"  All-States Optimal Value Accuracy: {best_metrics['all_value_acc']:.2f}% "
        f"(Fastest/Slowest Ply Accuracy: {best_metrics['all_ply_acc']:.2f}%)"
    )
    print(f"  Mean |Q_DQN(s,a) - Q*(s,a)| Error : {best_metrics['mean_q_mae']:.4f}")

    vs_random = play_match_series(agent, "random", oracle, num_games_per_side=200)
    vs_oracle = play_match_series(agent, "oracle", oracle, num_games_per_side=100)

    print("\nMatch Evaluation (200 games per side vs Random):")
    print(
        f"  DQN as Player 1: {vs_random['as_p1']['wins']}W / "
        f"{vs_random['as_p1']['draws']}D / {vs_random['as_p1']['losses']}L"
    )
    print(
        f"  DQN as Player 2: {vs_random['as_p2']['wins']}W / "
        f"{vs_random['as_p2']['draws']}D / {vs_random['as_p2']['losses']}L"
    )

    print("\nMatch Evaluation (100 games per side vs Perfect Oracle):")
    print(
        f"  DQN as Player 1: {vs_oracle['as_p1']['wins']}W / "
        f"{vs_oracle['as_p1']['draws']}D / {vs_oracle['as_p1']['losses']}L"
    )
    print(
        f"  DQN as Player 2: {vs_oracle['as_p2']['wins']}W / "
        f"{vs_oracle['as_p2']['draws']}D / {vs_oracle['as_p2']['losses']}L"
    )
    print("=" * 72)

    return agent, oracle, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Double Dueling DQN for Chopsticks.")
    parser.add_argument("--episodes", type=int, default=6000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dqn_chopsticks.pt",
        help="Path to save best model checkpoint",
    )
    args = parser.parse_args()
    train_dqn(
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
