import streamlit as st
from pypokerengine.api.game import setup_config, start_poker
from CFRBot import TreeCFRPlayer, RandomPlayer
import pandas as pd
import io
import sys
import re

# Title and Description
st.title("Counterfactual Regret Minimization Poker Bot")
st.markdown("""
A demonstration of **Counterfactual Regret Minimization (CFR)** for heads-up No Limit Texas Hold’em, 
integrated with [PyPokerEngine](https://github.com/ishikota/PyPokerEngine).

---

### Instructions:
1. Configure the game settings using the sidebar.
2. Click **Start Game** to watch the CFR bot play against a RandomPlayer.
3. View the action log and results below, organized by rounds.
4. Learn more about the code in the **Code Explanation** section.
""")

# Tabs: Game & Code Explanation
tabs = st.tabs(["Game", "Code Explanation"])

with tabs[0]:  # Game Tab
    st.sidebar.header("Game Configuration")
    num_rounds = st.sidebar.slider("Number of Rounds", min_value=1, max_value=50, value=10, step=1)
    initial_stack = st.sidebar.slider("Initial Stack Size", min_value=500, max_value=5000, value=1000, step=100)
    small_blind = st.sidebar.slider("Small Blind Amount", min_value=10, max_value=200, value=10, step=5)

    st.write("### Game Log by Round")
    if st.button("Start Game"):
        # Set up PyPokerEngine configuration
        config = setup_config(max_round=num_rounds, initial_stack=initial_stack, small_blind_amount=small_blind)
        config.register_player(name="CFR_BOT", algorithm=TreeCFRPlayer(train_mode=True))
        config.register_player(name="RandBot", algorithm=RandomPlayer())

        # Redirect verbose output to capture logs manually
        log_capture_string = io.StringIO()
        sys.stdout = log_capture_string

        # Start the game
        try:
            st.write("Starting game...")
            game_result = start_poker(config, verbose=1)
        finally:
            sys.stdout = sys.__stdout__  # Restore original stdout

        # Process logs into a structured format
        logs = log_capture_string.getvalue().splitlines()

        # Parse rounds and actions
        rounds = []
        current_round = None

        for line in logs:
            if "Started the round" in line:
                if current_round:
                    rounds.append(current_round)
                round_number = int(line.split()[-1])
                current_round = {"Round": round_number, "Actions": []}
            elif "declared" in line:
                # Parse player actions
                parts = line.split()
                player = parts[0].strip('"')
                action = parts[2].strip('"')
                # Extract chips using regex for patterns like "call:20" or "raise:50"
                chips_match = re.search(r":(\d+)", line)
                chips = chips_match.group(1) if chips_match else "0"  # Default to 0 for non-betting actions
                current_round["Actions"].append({"Player": player, "Action": action, "Chips": chips})
            elif "won the round" in line:
                # Parse round results
                winner = line.split("[")[1].split("]")[0]
                stack_info = line.split("stack = ")[1]
                current_round["Actions"].append({"Player": winner, "Action": "Won", "Chips": stack_info})

        if current_round:
            rounds.append(current_round)

        # Display each round in a table
        for round_info in rounds:
            st.write(f"### Round {round_info['Round']}")
            if round_info["Actions"]:
                actions_df = pd.DataFrame(round_info["Actions"])
                st.table(actions_df)
            else:
                st.write("No actions were recorded for this round.")

        st.write("### Final Results")
        for player in game_result["players"]:
            st.write(f"**{player['name']}** ended with **{player['stack']} chips**.")

with tabs[1]:  # Code Explanation Tab
    st.write("### Code Explanation")

    with st.expander("CFR (Counterfactual Regret Minimization) Algorithm"):
        st.markdown("""
        - **Core Idea**: CFR minimizes regret over many iterations to approximate an optimal strategy.
        - **Key Components**:
          - `CFRNode`: Stores regret and strategy sums for decision points.
          - `TreeNode`: Represents the decision tree structure for the game.
          - `cfr_tree()`: Recursive forward/backward pass to compute regrets and strategies.
        """)

    with st.expander("Tree-based Game Representation"):
        st.markdown("""
        - The game is modeled as a tree:
          - **Nodes**: Represent game states (e.g., decisions made by players).
          - **Edges**: Represent actions taken (e.g., "raise", "call").
        - Example:
          ```
          Root
          ├── Fold (Terminal)
          ├── Call
          │   ├── Opponent Fold (Terminal)
          │   ├── Opponent Call (Terminal)
          │   └── Opponent Raise (Terminal)
          └── Raise
              ├── Opponent Fold (Terminal)
              ├── Opponent Call (Terminal)
              └── Opponent Raise (Terminal)
          ```
        - This structure allows CFR to explore and compute strategies for each decision.
        """)

    with st.expander("Integration with PyPokerEngine"):
        st.markdown("""
        - **TreeCFRPlayer**:
          - Builds a decision tree for the current game state.
          - Runs a few iterations of CFR to compute optimal actions.
          - Chooses an action based on the average strategy.
        - **RandomPlayer**:
          - A baseline opponent that chooses actions randomly.
        """)

    with st.expander("Data Structures Used"):
        st.markdown("""
        - `CFRNode`:
          - Stores:
            - `regret_sums`: Tracks regret for each action.
            - `strategy_sums`: Tracks strategy probabilities over time.
          - Methods:
            - `get_strategy()`: Converts regrets to probabilities.
            - `get_average_strategy()`: Computes the time-averaged strategy.

        - `TreeNode`:
          - Represents a game state.
          - Stores:
            - `current_player`: The player whose turn it is.
            - `info_key`: Identifies the game state.
            - `children`: Actions and resulting nodes.
            - `terminal_payoff`: Final payoffs if the node is terminal.
        """)

    with st.expander("Example Workflow"):
        st.markdown("""
        1. **Game State**:
           - Players receive hole cards and community cards.
           - Current pot size and actions are noted.
        2. **Decision Tree**:
           - A small game tree is built for the current state.
           - Possible actions: ["fold", "call", "raise"].
        3. **CFR Execution**:
           - Runs a forward/backward pass on the tree.
           - Updates regrets and computes strategies.
        4. **Action Selection**:
           - The bot chooses an action based on the average strategy.
           - Action is returned to PyPokerEngine.
        5. **Repeat**:
           - For each decision point in the game, the above steps are repeated.
        """)
