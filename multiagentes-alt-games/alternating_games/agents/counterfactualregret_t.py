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
        # TODO
        actions = self.obs
        regrets = self.regrets(actions)
        regrets = np.array([np.maximum(regret, 0.0) for action, regret in enumerate(regrets)])
        
        total = regrets.sum()

        if total > 0:
            self.curr_policy = regrets / total
        else:
            self.curr_policy = np.ones_like(self.cum_regrets) / len(self.cum_regrets)

        self.sum_policy += self.curr_policy

    def regrets(self, played_actions: ActionDict) -> dict[AgentID, float]:
        actions = played_actions.copy()
        a = actions[self.agent]
        g = self.game.clone()
        u = np.zeros(g.num_actions(self.agent), dtype=float) # la utilidad de cada accion mia. 
        # 
        # TODO: calcular regrets
        #
        for action in range(g.num_actions(self.agent)):
            g_sim = self.game.clone()
            actions_sim = actions.copy()
            actions_sim[self.agent] = action
            g_sim.step(actions_sim)
            u[action] = g_sim.reward(self.agent)

        current_u = u[a]
        #print(g.num_actions(self.agent))
        #print(type(g.num_actions(self.agent)))
        r = u - current_u
        return r
    
    def update(self, utility, node_utility, probability) -> None:
        # update 
        # ...

        # regret matching policy
        self.regret_matching()  

    def policy(self):
        return self.learned_policy

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            a = np.argmax(np.random.multinomial(1, node.policy(), size=1))
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

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
        # TODO
        node_utility = 0 #remove

        return node_utility
        
