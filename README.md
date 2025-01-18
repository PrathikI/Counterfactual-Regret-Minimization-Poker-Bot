# Counterfactual-Regret-Minimization-Poker-Bot

A demonstration of **Counterfactual Regret Minimization (CFR)** for heads-up No Limit Texas Hold’em, integrated with [PyPokerEngine] (https://github.com/ishikota/PyPokerEngine).

## Key Points

- Implements a decision tree for Counterfactual Regret Minimization (CFR) in No Limit Hold’em.  
- Utilizes a custom `TreeCFRPlayer` to construct and train on a decision tree during gameplay, choosing actions based on the average CFR strategy.  
- Includes a `RandomPlayer` opponent for testing.  

## Features

- **CFR Logic**: Forward/backward pass implementation to update regrets and compute strategies.  
- **PyPokerEngine Integration**: Fully functional bot as a `BasePokerPlayer`.  
- **Action Logging**: Tracks `raise`, `call`, and `fold` decisions during gameplay for analysis.

## Running Instructions

- **Clone the Repository**: `git clone https://github.com/PrathikI/Counterfactual-Regret-Minimization-Poker-Bot.git`
- **Install Dependencies**: `pip install requirements.txt`
- **Run Frontend**: `streamlit run app.py`
