import itertools

import numpy as np
from tqdm import tqdm

import wandb
from algo.utils import DiscreFunc, WelfareFunc


class RAValueIteration:
    def __init__(self, env, discre_alpha, gamma, reward_dim, time_horizon, welfare_func_name, nsw_lambda, seed=1122, wdb=False) -> None:
        self.env = env
        self.discre_alpha = discre_alpha
        self.discre_func = DiscreFunc(discre_alpha)
        self.gamma = gamma
        self.reward_dim = reward_dim
        self.time_horizon = time_horizon
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda)
        self.training_complete = False
        self.seed = seed
        self.wdb = wdb
    
    def initialize(self):
        min, max = 0, np.floor(self.time_horizon / self.discre_alpha) * self.discre_alpha
        self.discre_grid = np.arange(min, max + self.discre_alpha, self.discre_alpha)
        self.init_Racc = [np.asarray(r) for r in list(itertools.product(self.discre_grid, repeat=self.reward_dim))]
        self.encode_Racc_dict = {str(r): i for i, r in enumerate(self.init_Racc)}
        
        # assert all([len(x) == self.reward_dim for x in self.init_Racc]), "Invalid reward accumulation"
        
        self.V = np.zeros((
            self.env.observation_space.n, 
            int(len(self.discre_grid) ** self.reward_dim),
            self.time_horizon + 1, 
        ))
        self.Pi = self.V.copy()
        
        for Racc in tqdm(self.init_Racc, desc="Initializing..."):
            for state in range(self.env.observation_space.n):
                Racc_code = self.encode_Racc(Racc)
                self.V[state, Racc_code, 0] = self.welfare_func(Racc)
    
    def encode_Racc(self, Racc):
        assert hasattr(self, "encode_Racc_dict"), "need to have initialize accumulated reward to begin with"
        return self.encode_Racc_dict[str(Racc)]
        
    def train(self):
        self.initialize()
        
        for t in tqdm(range(1, self.time_horizon + 1), desc="Training..."):
            min, max = 0, np.floor( (self.time_horizon - t) / self.discre_alpha ) * self.discre_alpha
            time_grid = np.arange(min, max + self.discre_alpha, self.discre_alpha)
            curr_Racc = [np.asarray(r) for r in list(itertools.product(time_grid, repeat=self.reward_dim))]
            # assert all([len(x) == self.reward_dim for x in curr_Racc]), "Invalid reward accumulation"
            
            for Racc in tqdm(curr_Racc, desc="Iterating Racc..."):
                Racc_code = self.encode_Racc(Racc)
                
                for state in range(self.env.observation_space.n):
                    transition_prob, reward_arr, next_state_arr = self.env.get_transition(state)        # vectorized, in dimensionality of the action space
                    
                    next_Racc = Racc + np.power(self.gamma, self.time_horizon - t) * reward_arr     # use broadcasting
                    next_Racc_discretized = self.discre_func(next_Racc)
                    next_Racc_code = [self.encode_Racc(d) for d in next_Racc_discretized]
                    
                    all_V = transition_prob * self.V[next_state_arr.astype(int), next_Racc_code, t - 1]
                    self.V[state, Racc_code, t] = np.max(all_V)
                    self.Pi[state, Racc_code, t] = np.argmax(all_V)
            
            self.evaluate()
        
        print("Finish training")
    
    def evaluate(self):
        self.env.seed(self.seed)
        state = self.env.reset()
        Racc = np.zeros(self.reward_dim)
        c = 0
        
        for t in range(self.time_horizon):
            Racc_code = self.encode_Racc(self.discre_func(Racc))
            action = self.Pi[state, Racc_code, t]
            
            next, reward, done = self.env.step(action)
            state = next
            Racc += np.power(self.gamma, c) * reward
            
            c += 1
        
        if self.wdb:
            wandb.log({"welfare": self.welfare_func(Racc)})
        