from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
from typing import Callable
from eval import default_eval

class MCTSNode:
    def __init__(self, parent: 'MCTSNode', game: AlternatingGame, action: ActionType):
        self.parent = parent
        self.game = game
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
        self.cum_rewards = np.zeros(len(game.agents))
        self.agent = self.game.agent_selection

def ucb(node, C=sqrt(2)) -> float:
    agent_idx = node.game.agent_name_mapping[node.agent]
    if node.visits==0:
        return float('inf')
    return node.cum_rewards[agent_idx] / node.visits + C * sqrt(log(node.parent.visits)/node.visits)

def uct(node: MCTSNode, agent: AgentID) -> MCTSNode:
    child = max(node.children, key=ucb)
    return child

class MonteCarloTreeSearch(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, simulations: int=500, rollouts: int=10, selection: Callable[[MCTSNode, AgentID], MCTSNode]=uct, eval: Callable[[MCTSNode, AgentID], float] = default_eval) -> None:
        """
        Parameters:
            game: alternating game associated with the agent
            agent: agent id of the agent in the game
            simulations: number of MCTS simulations (default: 100)
            rollouts: number of MC rollouts (default: 10)
            selection: tree search policy (default: uct)
        """
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.selection = selection
        self.eval = eval
        
    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a

    def mcts(self) -> (ActionType, float):

        root = MCTSNode(parent=None, game=self.game, action=None)

        for i in range(self.simulations):

            node = root
            node.game = self.game.clone()

            #print(i)
            #node.game.render()

            # selection
            #print('selection')
            node = self.select_node(node=node)

            # expansion
            #print('expansion')
            self.expand_node(node)

            # rollout
            #print('rollout')
            rewards = self.rollout(node)

            #update values / Backprop
            #print('backprop')
            self.backprop(node, rewards)

        #print('root childs')
        #for child in root.children:
        #    print(child.action, child.cum_rewards / child.visits)

        action, value = self.action_selection(root)

        return action, value

    def backprop(self, node, rewards):
        # TODO
        # cumulate rewards and visits from node to root navigating backwards through parent
        while node is not None:
            node.visits += 1
            node.cum_rewards += rewards
            node = node.parent
            # If node is None, we have reached the root

    def rollout(self, node):
        rewards = np.zeros(len(self.game.agents))
        # TODO
        # implement rollout policy
        for i in range(self.rollouts): 
        #     play random game and record average rewards
            game = node.game.clone()
            while not game.terminated():
                actions = game.available_actions()
                if not actions:
                    break
                action = np.random.choice(actions)
                game.step(action)
            # accumulate rewards for each agent
            for agent in game.agents:
                agent_idx = game.agent_name_mapping[agent]
                if game.terminated():
                    # If game is terminated, we can get the rewards
                    rewards[agent_idx] += game.rewards[agent]
                else:
                    # If game is not terminated, we can use the evaluation function
                    rewards[agent_idx] += self.eval(game, agent)
        rewards /= self.rollouts  # Average rewards over rollouts
        return rewards

    def select_node(self, node: MCTSNode) -> MCTSNode:
        curr_node = node
        while curr_node.children:
            if curr_node.explored_children < len(curr_node.children):
                # TODO
                # set curr_node to an unvisited child
                idx = curr_node.explored_children
                curr_node.explored_children += 1
                curr_node = curr_node.children[idx]    
            else:
                # TODO
                # set curr_node to a child using the selection function
                #curr_node = self.selection(curr_node, curr_node.agent)
                curr_node = self.selection(curr_node, self.agent)
        return curr_node

    def expand_node(self, node) -> None:
        # TODO
        # if the game is not terminated: 
        #    play an available action in node
        #    create a new child node and add it to node children
        #if not node.game.terminated():
        if node.game.terminated():
            return
        for action in node.game.available_actions():
            # np.random.shuffle(actions)
            # if actions:
            #     action = np.random.choice(actions)
            #     child_game = node.game.clone()
            #     child_game.step(action)
            #     # Create a new child node
            #     child_node = MCTSNode(parent=node, game=child_game, action=action)
            #     # Initialize child node
            #     node.children.append(child_node)
            #     #node.explored_children += 1
            g = node.game.clone()
            g.step(action)
            node.children.append(MCTSNode(node, g, action))#, self.agent))

    def action_selection(self, node: MCTSNode) -> (ActionType, float):
        action: ActionType = None
        value: float = 0
        # TODO
        # hint: return action of child with max value 
        # other alternatives could be considered
        if node.children:
            best_child = max(node.children, key=lambda child: child.cum_rewards[child.game.agent_name_mapping[self.agent]] / child.visits)
            action = best_child.action
            value = best_child.cum_rewards[best_child.game.agent_name_mapping[self.agent]] / best_child.visits
            # If no children, return None or some default value
        return action, value    