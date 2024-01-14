"""
Implementation of mixture policy.

Reference
---------
    Fan, Z., Peng, N., Tian, M., & Fain, B. Welfare and Fairness in Multi-objective Reinforcement Learning.
    https://github.com/MuhangTian/Fair-MORL-AAMAS
"""
import numpy as np
from algo.utils import WelfareFunc
import wandb

class MixturePolicy:
    def __init__(self, env, episodes, time_horizon, lr, epsilon, gamma, init_val, weights, interval, welfare_func_name, save_path, nsw_lambda, p=None, seed=2023, wdb=False):
        self.env = env
        self.episodes = episodes
        self.welfare_func_name = "nash welfare" if welfare_func_name == "nash-welfare" else welfare_func_name
        self.time_horizon = time_horizon
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_val = init_val
        self.weights = weights
        self.interval = interval
        self.save_path = save_path
        self.seed = seed
        self.wdb = wdb
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p)
        self.dim = len(self.env.loc_coords)
        self.Racc_record = []
        
    def greedy(self, vec, weights):
        '''Helper function'''
        arr = []
        for val in vec: 
            arr.append(np.dot(weights, val))    # linear scalarization
        return np.argmax(arr)
    
    def initialize(self):
        self.dims = [i for i in range(len(self.weights))]

        self.policies = []
        for dim in self.dims:    # Obtain set of policies
            q = np.full([self.env.observation_space.n, self.env.action_space.n, self.dim], self.init_val, dtype=float)
            self.policies.append(q)
        
    def train(self):
        self.initialize()

        for i in range(1, self.episodes+1):
            R_acc = np.zeros(self.dim)
            state = self.env.reset()
            done = False
            count, dim, c = 0, 0, 0
            Q = self.policies[dim]
            weights = self.weights[dim]
            
            while not done:
                if count > int(self.time_horizon/self.dim/self.interval):   # determines the period of changing policies
                    dim += 1
                    if dim >= self.dim: 
                        dim = 0  # back to first objective after a "cycle"
                    Q = self.policies[dim]
                    weights = self.weights[dim]
                    count = 0   # change policy after t/d timesteps
                    
                if np.random.uniform(0, 1) < self.epsilon: 
                    action = self.env.action_space.sample()
                else: 
                    action = self.greedy(Q[state], weights)
                
                next, reward, done = self.env.step(action)
                count += 1
                next_action = self.greedy(Q[next], weights)
                
                for j in range(len(Q[state, action])):
                    Q[state,action][j] = Q[state,action][j] + self.lr*(reward[j]+self.gamma*Q[next,next_action][j]-Q[state,action][j])

                state = next
                R_acc += np.power(self.gamma, c)*reward
                c += 1
            
            R_acc = np.where(R_acc < 0, 0, R_acc) # Replace the negatives with 0
            
            if self.welfare_func_name == "nash welfare":
                nonlinear_score = np.power(np.product(R_acc), 1/len(R_acc))
            elif self.welfare_func_name in ["p-welfare", "egalitarian"]:
                nonlinear_score = self.welfare_func(R_acc)
            else:
                raise ValueError("Invalid welfare function name")
            
            self.Racc_record.append(R_acc)
            print(f"R_acc: {R_acc}, {self.welfare_func_name}: {nonlinear_score}")
            
            if self.wdb:
                wandb.log({self.welfare_func_name: nonlinear_score})
        
        print("Finish training")
        np.savez(self.save_path, policies=self.policies, Racc_record=np.asarray(self.Racc_record)) 