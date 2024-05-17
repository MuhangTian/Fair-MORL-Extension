import itertools

import numpy as np
from tqdm import tqdm
from sys import getsizeof

import wandb
from algo.utils import DiscreFunc, WelfareFunc


class RAValueIteration:
    def __init__(self, env, discre_alpha, growth_rate, gamma, reward_dim, time_horizon, welfare_func_name, nsw_lambda, save_path, seed=1122, p=None, threshold=5, wdb=False, scaling_factor=1) -> None:
        self.env = env
        self.welfare_func_name = "nash welfare" if welfare_func_name == "nash-welfare" else welfare_func_name
        self.discre_alpha = discre_alpha # Discretization factor for accumulated rewards.
        self.discre_func = DiscreFunc(discre_alpha, growth_rate)  # Discretization function with growth rate for exponential discretization.
        self.gamma = gamma
        self.scaling_factor = scaling_factor
        self.reward_dim = reward_dim
        self.time_horizon = time_horizon
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p, threshold)
        self.training_complete = False
        self.seed = seed
        self.wdb = wdb
        self.save_path = save_path
    
    def initialize(self):
        # min, max = 0, np.floor(self.time_horizon / self.discre_alpha) * self.discre_alpha
        # self.discre_grid = np.arange(min, max + self.discre_alpha, self.discre_alpha)

        # Calculate maximum possible reward accumulation for exponential discretization
        if self.gamma == 1:  # Special case where gamma is 1, just multiply max reward by the number of steps
            max_reward = self.time_horizon/self.scaling_factor
        else:
            # Calculate the sum of the geometric series
            sum_of_discounts = (1 - self.gamma ** (self.time_horizon/self.scaling_factor)) / (1 - self.gamma)
            max_reward = sum_of_discounts
        max_discrete = self.discre_func(max_reward)

        print(f"Max discretized reward during initialization: {max_discrete}")

        self.discre_grid = np.unique(np.array([self.discre_func(alpha * self.discre_alpha) for alpha in range(int(np.ceil(max_discrete / self.discre_alpha)))]))
        self.init_Racc = [np.asarray(r) for r in list(itertools.product(self.discre_grid, repeat=self.reward_dim))] # All possible reward accumulations.
        self.encode_Racc_dict = {str(r): i for i, r in enumerate(self.init_Racc)} # Encoding of reward accumulations.
        
        # assert all([len(x) == self.reward_dim for x in self.init_Racc]), "Invalid reward accumulation"
        
        self.V = np.zeros((
            self.env.observation_space.n, 
            int(len(self.discre_grid) ** self.reward_dim),
            self.time_horizon + 1, 
        ))
        self.Pi = self.V.copy()
        
        if self.wdb:
            wandb.log({
                "memory (mb)": round(getsizeof(self.V) / 1024 / 1024, 2),
                "state_space_size": self.env.observation_space.n,
                "discre_grid_size": len(self.discre_grid)
            })
        
        for Racc in tqdm(self.init_Racc, desc="Initializing..."):
            for state in range(self.env.observation_space.n):
                Racc_code = self.encode_Racc(Racc)
                self.V[state, Racc_code, 0] = self.welfare_func(Racc)
    
    def encode_Racc(self, Racc):
        # Encode the accumulated reward for indexing.
        assert hasattr(self, "encode_Racc_dict"), "need to have initialize accumulated reward to begin with"
        return self.encode_Racc_dict[str(Racc)]
        
    def train(self):
        self.initialize()
        
        for t in tqdm(range(1, self.time_horizon + 1), desc="Training..."):
            # min, max = 0, np.floor( (self.time_horizon - t) / self.discre_alpha ) * self.discre_alpha
            # time_grid = np.arange(min, max + self.discre_alpha, self.discre_alpha)

            # Use the discretized grid from initialization that accounts for exponential growth
            # Calculate the sum of the geometric series for the remaining timesteps
            remaining_steps = self.time_horizon - t
            max_accumulated_discounted_reward = (1 - self.gamma ** (remaining_steps)) / (1 - self.gamma)

            max_possible_reward = min(max_accumulated_discounted_reward, self.discre_grid[-2])
            
            # max_possible_reward = np.round(max_accumulated_discounted_reward / self.discre_alpha) * self.discre_alpha / scaling_factor

            max_possible_discretized_reward = self.discre_func(max_possible_reward)

            print()
            print(f"Max discretized reward at t = {remaining_steps}: {max_possible_discretized_reward}")
            
            # Filter the discretization grid based on the computed max possible reward
            time_grid = self.discre_grid[self.discre_grid <= max_possible_discretized_reward]
            curr_Racc = [np.asarray(r) for r in list(itertools.product(time_grid, repeat=self.reward_dim))] # Current possible rewards.
            # assert all([len(x) == self.reward_dim for x in curr_Racc]), "Invalid reward accumulation"

            for Racc in tqdm(curr_Racc, desc="Iterating Racc..."):
                Racc_code = self.encode_Racc(Racc)
                
                for state in range(self.env.observation_space.n):
                    transition_prob, reward_arr, next_state_arr = self.env.get_transition(state)        # vectorized, in dimensionality of the action space
                    
                    next_Racc = Racc + np.power(self.gamma, self.time_horizon - t) * reward_arr     # use broadcasting
                    next_Racc_discretized = self.discre_func(next_Racc) # Discretize next rewards.
                    next_Racc_code = [self.encode_Racc(d) for d in next_Racc_discretized]
                    
                    all_V = transition_prob * self.V[next_state_arr.astype(int), next_Racc_code, t - 1]
                    self.V[state, Racc_code, t] = np.max(all_V)
                    self.Pi[state, Racc_code, t] = np.argmax(all_V)
            
            # self.evaluate()
        
        print("Finish training")
        self.evaluate(final=True)
        np.savez(self.save_path, V=self.V, Pi=self.Pi, Racc_record=self.Racc_record)
    
    def evaluate(self, final=False):
        self.env.seed(self.seed)
        state = self.env.reset()
        Racc = np.zeros(self.reward_dim)
        c = 0
        
        for t in range(self.time_horizon, 0, -1):
            Racc_code = self.encode_Racc(self.discre_func(Racc))
            action = self.Pi[state, Racc_code, t]
            
            next, reward, done = self.env.step(action)
            self.env.render()
            decoded_pos, decoded_status = self.env.decode_state(next)
            print(f"Decoded Position: {decoded_pos}, Decoded Resource Status: {decoded_status}")
            state = next
            Racc += np.power(self.gamma, c) * reward
            print(f"Accumulated Reward: {Racc}")
            
            c += 1
        
        if self.welfare_func_name == "nash welfare":
            if self.wdb:
                wandb.log({self.welfare_func_name: self.welfare_func.nash_welfare(Racc)})
            print(f"{self.welfare_func_name}: {self.welfare_func.nash_welfare(Racc)}, Racc: {Racc}")
        elif self.welfare_func_name in ["p-welfare", "egalitarian", "resource_damage_scalarization"]:
            if self.wdb:
                wandb.log({self.welfare_func_name: self.welfare_func(Racc)})
            print(f"{self.welfare_func_name}: {self.welfare_func(Racc)}, Racc: {Racc}")
            
        if final:
            self.Racc_record = Racc
        