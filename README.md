# Chopsticks DQN

Action-Masked Double Dueling Deep Q-Network (DQN) and Exact Retrograde Minimax Oracle for the classic [Chopsticks](https://en.wikipedia.org/wiki/Chopsticks_(hand_game)) hand game.

## Rules Implemented
- **Rollover (Modulo 5)**: Attacking a hand adds fingers modulo 5 (`(attacker + target) % 5`); reaching exactly `5` eliminates the hand (`0`).
- **Division-Only Split**: When one hand is dead (`0`) and the live hand has an even number of fingers (`2` or `4`), the player can divide equally (`(2, 0) -> (1, 1)` or `(4, 0) -> (2, 2)`).
- **Termination**:
  - **Win / Loss**: Both hands of a player reach `0`.
  - **Draw**: 3-fold canonical state repetition or `max_steps = 100`.

## Quick Start (with `uv` and Python 3.14)

```powershell
# Install dependencies
uv sync --all-extras

# Run unit tests
uv run pytest -v

# Train the DQN and benchmark against the Retrograde Oracle
uv run python -m chopsticks.train

# Play interactively against the trained DQN or Oracle
uv run python -m chopsticks.cli
```
