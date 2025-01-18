# Counterfactual-Regret-Minimization-Poker-Bot
A concise demonstration of Counterfactual Regret Minimization (CFR) for heads-up No Limit Texas Hold’em, integrated with PyPokerEngine (https://github.com/ishikota/PyPokerEngine).

## Key Points

- Demonstrates a _small, illustrative_ decision tree for CFR, rather than a full-blown No-Limit Hold’em solver.
- Uses a custom `TreeCFRPlayer` that builds a toy game tree each time it acts, runs a few CFR iterations, then chooses an action from the average strategy.
- Includes a `RandomPlayer` opponent for basic testing.

## Features

- **Tree-based CFR**: Minimal forward/backward pass logic to update regrets.
- **PyPokerEngine** integration: Works as a `BasePokerPlayer`.
- **Two-Item Return**: Satisfies PyPokerEngine’s requirement to return `(action, amount)`.

## Setup & Installation

