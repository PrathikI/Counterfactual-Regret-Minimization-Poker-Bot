import random
import numpy as np
from collections import defaultdict

try:
    from pypokerengine.api.game import setup_config, start_poker
    from pypokerengine.players import BasePokerPlayer
except ImportError:
    raise ImportError("PyPokerEngine is not installed. Please install via 'pip install pypokerengine'")

############################################
# 1) CFR Data Structures + Global Dictionary
############################################

class CFRNode:
    """
    A node storing regret sums and strategy sums for a particular information set (info_key).
    """
    def __init__(self, info_key, actions):
        self.info_key = info_key
        self.actions = actions
        self.regret_sums = defaultdict(float)
        self.strategy_sums = defaultdict(float)

    def get_strategy(self, realization_weight):
        """
        Convert regret sums into a probability distribution over actions,
        then update strategy_sums for average-strategy tracking.
        """
        positive_regrets = {a: max(0.0, self.regret_sums[a]) for a in self.actions}
        sum_reg = sum(positive_regrets.values())

        if sum_reg > 1e-9:
            strategy = {a: positive_regrets[a]/sum_reg for a in self.actions}
        else:
            strategy = {a: 1.0/len(self.actions) for a in self.actions}

        for a in strategy:
            self.strategy_sums[a] += strategy[a] * realization_weight
        return strategy

    def get_average_strategy(self):
        total = sum(self.strategy_sums.values())
        if total < 1e-9:
            return {a: 1.0/len(self.actions) for a in self.actions}
        return {a: self.strategy_sums[a]/total for a in self.actions}


class TreeNode:
    """
    Toy decision tree node with:
      - current_player: index (0 or 1) or None if terminal
      - info_key: identifies the situation (e.g. (hole_str, board_str, pot_size))
      - children: dict(action->TreeNode)
      - terminal_payoff: e.g. [p0_payoff, p1_payoff] if terminal
    """
    def __init__(self, current_player, info_key, children=None, terminal_payoff=None):
        self.current_player = current_player
        self.info_key = info_key
        self.children = children if children else {}
        self.terminal_payoff = terminal_payoff

    def is_terminal(self):
        return self.terminal_payoff is not None

# A global dictionary for storing CFR info sets
cfr_nodes = {}

def get_or_create_cfr_node(info_key, actions):
    if info_key not in cfr_nodes:
        cfr_nodes[info_key] = CFRNode(info_key, actions)
    return cfr_nodes[info_key]

############################################
# 2) Building a Small Decision Tree
############################################

def build_subtree(round_state, player_view, known_hole_cards, known_board, pot_size):
    """
    Creates a minimal tree with:
      "fold", "call", "raise"
    plus a tiny "opponent node" if we call/raise.
    """
    hole_str = "-".join(sorted([c[0] for c in known_hole_cards]))
    board_str = "-".join([c[0] for c in known_board])
    info_key = (hole_str, board_str, pot_size)

    actions = ["fold", "call", "raise"]
    root_node = TreeNode(current_player=player_view, info_key=info_key)

    # Fold => hero loses 10
    fold_payoff = [0, 0]
    fold_payoff[player_view] = -10
    fold_child = TreeNode(None, None, terminal_payoff=fold_payoff)

    # Opponent node
    opp_view = 1 - player_view
    opp_info_key = ("opp_turn", pot_size)
    opp_node = TreeNode(current_player=opp_view, info_key=opp_info_key)

    # Opponent actions => terminal payoffs
    fold_opp_payoff = [0, 0]
    fold_opp_payoff[player_view] = +25
    fold_opp_child = TreeNode(None, None, terminal_payoff=fold_opp_payoff)

    call_opp_payoff = [0, 0]
    call_opp_payoff[player_view] = +5
    call_opp_payoff[opp_view] = -5
    call_opp_child = TreeNode(None, None, terminal_payoff=call_opp_payoff)

    raise_opp_payoff = [0, 0]
    raise_opp_payoff[player_view] = -15
    raise_opp_payoff[opp_view] = +15
    raise_opp_child = TreeNode(None, None, terminal_payoff=raise_opp_payoff)

    # Link them
    opp_node.children["fold_opp"] = fold_opp_child
    opp_node.children["call_opp"] = call_opp_child
    opp_node.children["raise_opp"] = raise_opp_child

    root_node.children["fold"] = fold_child
    root_node.children["call"] = opp_node
    root_node.children["raise"] = opp_node

    return root_node

############################################
# 3) CFR Recursion on the Sub-Tree
############################################

def cfr_tree(node, reach_probs):
    """
    Forward/backward pass for CFR.
    Returns payoff array, e.g. [u0,u1].
    """
    if node.is_terminal():
        return node.terminal_payoff

    cp = node.current_player
    info_key = node.info_key
    actions = list(node.children.keys())

    cfr_node = get_or_create_cfr_node(info_key, actions)
    strategy = cfr_node.get_strategy(reach_probs[cp])

    action_utils = {}
    node_util = np.zeros(2)
    for a in actions:
        child = node.children[a]
        next_reach = reach_probs[:]
        next_reach[cp] *= strategy[a]

        util = cfr_tree(child, next_reach)
        action_utils[a] = util
        node_util += strategy[a] * np.array(util)

    # Regret update
    cp_util = node_util[cp]
    for a in actions:
        r = action_utils[a][cp] - cp_util
        cfr_node.regret_sums[a] += r * reach_probs[cp]

    return node_util

def train_cfr_on_subtree(root_node, iterations=100):
    for _ in range(iterations):
        cfr_tree(root_node, [1.0, 1.0])

############################################
# 4) PyPokerEngine Players
############################################

class TreeCFRPlayer(BasePokerPlayer):
    def __init__(self, train_mode=True):
        super().__init__()
        self.train_mode = train_mode
        self.uuid = None

    def receive_game_start_message(self, game_info):
        self.uuid = self.uuid

    def receive_round_start_message(self, round_count, hole_cards, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def declare_action(self, valid_actions, hole_cards, round_state):
        pot_size = round_state["pot"]["main"]["amount"]
        community_cards = round_state["community_card"]
        seats = round_state["seats"]

        # Which seat am I?
        player_view = 0
        for i, seat_info in enumerate(seats):
            if seat_info["uuid"] == self.uuid:
                player_view = i
                break

        # Build sub-tree
        root_node = build_subtree(round_state, player_view, hole_cards, community_cards, pot_size)

        # Train
        if self.train_mode:
            train_cfr_on_subtree(root_node, iterations=50)

        # Retrieve average strategy from root
        hole_str = "-".join(sorted([c[0] for c in hole_cards]))
        board_str = "-".join([c[0] for c in community_cards])
        root_info_key = (hole_str, board_str, pot_size)

        if root_info_key not in cfr_nodes:
            strategy = {"fold":0.33, "call":0.34, "raise":0.33}
        else:
            node = cfr_nodes[root_info_key]
            strategy = node.get_average_strategy()

        return self._map_strategy_to_action(valid_actions, strategy)

    def _map_strategy_to_action(self, valid_actions, strategy):
        filtered_strat = {}
        sum_prob = 0.0

        for va in valid_actions:
            act_type = va["action"]
            if act_type in ("call", "check"):
                prob = strategy.get("call", 0.0)
                filtered_strat[act_type] = prob
                sum_prob += prob
            elif act_type == "raise":
                prob = strategy.get("raise", 0.0)
                filtered_strat["raise"] = prob
                sum_prob += prob
            elif act_type == "fold":
                prob = strategy.get("fold", 0.0)
                filtered_strat["fold"] = prob
                sum_prob += prob

        if sum_prob < 1e-9:
            # fallback
            choice = random.choice(valid_actions)
            return self._convert_to_action_amount(choice)

        for k in filtered_strat:
            filtered_strat[k] /= sum_prob

        r = random.random()
        cumulative = 0.0
        for va in valid_actions:
            act_type = va["action"]
            if act_type in ("call","check"):
                prob = filtered_strat.get("call", 0.0)
            elif act_type == "raise":
                prob = filtered_strat.get("raise", 0.0)
            else:
                prob = filtered_strat.get("fold", 0.0)

            cumulative += prob
            if r < cumulative:
                return self._convert_to_action_amount(va)

        return self._convert_to_action_amount(valid_actions[-1])

    def _convert_to_action_amount(self, va):
        act_type = va["action"]
        if act_type == "fold":
            return ("fold", 0)
        elif act_type in ("call","check"):
            if isinstance(va["amount"], dict):
                return ("call", 0)
            else:
                return ("call", va["amount"])
        elif act_type == "raise":
            return ("raise", va["amount"]["min"])
        return ("fold", 0)


class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_cards, round_state):
        choice = random.choice(valid_actions)
        return self._convert_to_two_tuple(choice)

    def _convert_to_two_tuple(self, va):
        act_type = va["action"]
        if act_type == "fold":
            return ("fold", 0)
        elif act_type in ("call","check"):
            # call or check => ("call", amount or 0)
            if isinstance(va["amount"], dict):
                return ("call", 0)
            else:
                return ("call", va["amount"])
        elif act_type == "raise":
            return ("raise", va["amount"]["min"])
        return ("fold", 0)

    def receive_game_start_message(self, game_info):
        pass
    def receive_round_start_message(self, round_count, hole_cards, seats):
        pass
    def receive_street_start_message(self, street, round_state):
        pass
    def receive_game_update_message(self, action, round_state):
        pass
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


############################################
# 7) Demo: Running a Game
############################################

def demo_run_game(num_rounds=5, train_mode=True):
    config = setup_config(
        max_round=num_rounds,
        initial_stack=1000,
        small_blind_amount=10
    )

    config.register_player(name="CFR_BOT", algorithm=TreeCFRPlayer(train_mode=train_mode))
    config.register_player(name="RandBot", algorithm=RandomPlayer())

    game_result = start_poker(config, verbose=1)
    print("\n=== Game finished. Results ===")
    for pl in game_result["players"]:
        print(f"{pl['name']} ended with {pl['stack']}")

if __name__ == "__main__":
    cfr_nodes = {}
    demo_run_game(num_rounds=10, train_mode=True)