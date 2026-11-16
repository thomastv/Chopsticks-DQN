"""Interactive CLI for playing Chopsticks against the trained DQN or Retrograde Oracle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chopsticks.dqn import DQNAgent
from chopsticks.env import ACTION_NAMES, NUM_ACTIONS, ChopsticksEnv, State
from chopsticks.oracle import RetrogradeOracle


def format_hand(fingers: int) -> str:
    """Return an ASCII visual representation of a hand's finger count."""
    if fingers == 0:
        return "[  DEAD (0)  ]"
    bars = " | " * fingers
    return f"[{bars:^12}] ({fingers})"


def print_board(env: ChopsticksEnv, p0_label: str, p1_label: str) -> None:
    """Display the current game board in the terminal."""
    (p0_l, p0_r), (p1_l, p1_r) = env.get_player_hands()
    turn_marker_0 = ">>" if env.current_player == 0 else "  "
    turn_marker_1 = ">>" if env.current_player == 1 else "  "

    print("\n" + "-" * 64)
    print(f"Step {env.steps:3d} | Active Turn: Player {env.current_player + 1}")
    print(f"  {turn_marker_1} P2 ({p1_label:8s}) : L {format_hand(p1_l)}   R {format_hand(p1_r)}")
    print(f"  {turn_marker_0} P1 ({p0_label:8s}) : L {format_hand(p0_l)}   R {format_hand(p0_r)}")
    print("-" * 64)


def print_action_analysis(
    state: State,
    legal_mask: np.ndarray,
    agent: DQNAgent | None,
    oracle: RetrogradeOracle,
) -> None:
    """Print a table of legal moves with DQN Q-values and exact Oracle evaluations."""
    q_vals = agent.get_q_values(state, mask_illegal=True) if agent is not None else None
    outcome_labels = {1: "WIN (+1)", 0: "DRAW (0)", -1: "LOSS (-1)"}

    print("  ID | Action           | DQN Q(s,a) | Oracle Verdict | Exact Q*(s,a)")
    print("  ---+------------------+------------+----------------+--------------")
    for a in range(NUM_ACTIONS):
        if not legal_mask[a]:
            continue
        outcome, ply = oracle.action_outcome(state, a)
        verdict = outcome_labels[outcome]
        if outcome != 0:
            verdict += f" in {ply}p"
        exact_q = oracle.exact_discounted_q(state, a, gamma=agent.gamma if agent else 0.95)
        q_str = f"{q_vals[a]:+8.4f}" if q_vals is not None else "   N/A  "
        print(f"  [{a}] | {ACTION_NAMES[a]:<16s} |  {q_str}  | {verdict:<14s} |   {exact_q:+8.4f}")


def run_match(
    p0_type: str,
    p1_type: str,
    agent: DQNAgent,
    oracle: RetrogradeOracle,
    show_hints: bool = True,
) -> None:
    """Run a single interactive or automated match between `p0_type` and `p1_type`."""
    env = ChopsticksEnv(max_steps=100, repetition_limit=3)
    state, legal_mask = env.reset(starting_player=0)
    labels = {0: p0_type.upper(), 1: p1_type.upper()}
    player_types = {0: p0_type, 1: p1_type}

    while not env.done:
        print_board(env, labels[0], labels[1])
        cur = env.current_player
        p_type = player_types[cur]

        if show_hints or p_type != "human":
            print_action_analysis(state, legal_mask, agent, oracle)

        if p_type == "human":
            legal_indices = [int(i) for i in np.flatnonzero(legal_mask)]
            while True:
                raw = input(f"\nYour move {legal_indices} (or 'q' to quit): ").strip().lower()
                if raw in ("q", "quit", "exit"):
                    print("Exiting match.")
                    return
                if raw.isdigit() and int(raw) in legal_indices:
                    action = int(raw)
                    break
                print(f"Invalid choice. Please enter one of {legal_indices}.")
        elif p_type == "dqn":
            action = agent.select_action(state, legal_mask, epsilon=0.0)
            print(f"\n-> DQN chooses [{action}] {ACTION_NAMES[action]}")
        elif p_type == "oracle":
            action = oracle.select_action(state)
            print(f"\n-> Oracle chooses [{action}] {ACTION_NAMES[action]}")
        else:
            raise ValueError(f"Unknown player type: {p_type}")

        state, _, _, info = env.step(action)
        legal_mask = info["legal_mask"]  # type: ignore[assignment]

    print_board(env, labels[0], labels[1])
    if env.winner is not None:
        print(
            f"\n*** GAME OVER: Player {env.winner + 1} ({labels[env.winner]}) WINS "
            f"by {env.termination_reason}! ***"
        )
    else:
        print(f"\n*** GAME OVER: DRAW by {env.termination_reason}! ***")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Chopsticks against DQN or Oracle.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dqn_chopsticks.pt",
        help="Path to trained DQN checkpoint",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a non-interactive exhibition match (DQN vs Oracle) and exit",
    )
    args = parser.parse_args()

    oracle = RetrogradeOracle()
    agent = DQNAgent(gamma=0.95)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        agent.load(ckpt_path)
        print(f"Loaded trained DQN checkpoint from {ckpt_path}")
    else:
        print(
            f"Warning: Checkpoint {ckpt_path} not found. Using untrained DQN weights. "
            "Run `uv run python -m chopsticks.train` first!"
        )

    if args.demo:
        print("\n=== Exhibition Match: DQN (P1) vs. Oracle (P2) ===")
        run_match("dqn", "oracle", agent, oracle, show_hints=True)
        return

    while True:
        print("\n=== Chopsticks Hand Game (Rollover Mod 5 + Division Split) ===")
        print("  1. Human (P1) vs. DQN (P2)")
        print("  2. DQN (P1)   vs. Human (P2)")
        print("  3. Human (P1) vs. Perfect Oracle (P2)")
        print("  4. Watch DQN (P1) vs. Perfect Oracle (P2)")
        print("  5. Quit")
        choice = input("Select mode [1-5]: ").strip()
        if choice == "1":
            run_match("human", "dqn", agent, oracle, show_hints=True)
        elif choice == "2":
            run_match("dqn", "human", agent, oracle, show_hints=True)
        elif choice == "3":
            run_match("human", "oracle", agent, oracle, show_hints=True)
        elif choice == "4":
            run_match("dqn", "oracle", agent, oracle, show_hints=True)
        elif choice in ("5", "q", "quit", "exit"):
            break
        else:
            print("Invalid selection.")


if __name__ == "__main__":
    main()
