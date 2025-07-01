import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent

class Node():

    def __init__(self, game: AlternatingGame, obs: ObsType) -> None:
        self.game = game
        self.agent = game.agent_selection
        self.obs = obs
        self.num_actions = self.game.num_actions(self.agent)
        self.cum_regrets = np.zeros(self.num_actions)
        self.curr_policy = np.full(self.num_actions, 1/self.num_actions)
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1
    
    def regret_matching(self):
        positive_regrets = np.maximum(self.cum_regrets, 0.0)
        total = positive_regrets.sum()

        if total > 0:
            self.curr_policy = positive_regrets / total
        else:
            self.curr_policy = np.ones_like(self.cum_regrets) / len(self.cum_regrets)

        self.sum_policy += self.curr_policy
    
    def update(self, utility, node_utility, probability):
        p = self.agent
        p_idx = self.game.agent_name_mapping[p]
        
        prod = 1
        for q in range(len(probability)):
            if q != p_idx:
                prod *= probability[q]

        self.cum_regrets += prod * (utility - node_utility)

        self.sum_policy += probability[p_idx] * self.curr_policy
        self.learned_policy = self.sum_policy

        self.regret_matching()  

    def policy(self):
        total = np.sum(self.learned_policy)
        if total > 0:
            return self.learned_policy / total
        else:
            return np.ones_like(self.learned_policy) / len(self.learned_policy)

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.env = game
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            a = np.argmax(np.random.multinomial(1, node.policy(), size=1))
            print(f'Agent {self.agent} plays action {a} with policy {node.policy()}')
            return a
        except:
            #raise ValueError('Train agent before calling action()')
            print('Node does not exist. Playing random.')
            return np.random.choice(self.game.available_actions())
    
    def train(self, niter=1000):
        for _ in range(niter):
            _ = self.cfr()

    def cfr(self):
        game = self.game.clone()
        utility: dict[AgentID, float] = dict()
        for agent in self.game.agents:
            game.reset()
            probability = np.ones(game.num_agents)
            utility[agent] = self.cfr_rec(game=game, agent=agent, probability=probability)

        return utility 

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: np.ndarray):
        if game.game_over():
            return game.reward(agent)
        
        q = game.agent_selection 
        info_set = game.observe(q)

        if info_set not in self.node_dict:
            self.node_dict[info_set] = Node(game.clone(), info_set)
        
        node = self.node_dict[info_set]
        legal_actions = list(game.action_iter(q))
        u = np.zeros(len(legal_actions))
        v = 0

        for i, action in enumerate(legal_actions):
            game_clone = game.clone()
            game_clone.step(action)

            prob_copy = probability.copy()
            q_idx = game.agent_name_mapping[q]
            prob_copy[q_idx] *= node.curr_policy[i]

            u[i] = self.cfr_rec(game_clone, agent, prob_copy)
            v += node.curr_policy[i] * u[i]

        if q == agent:
            node.update(u, v, probability)

        return v
        
