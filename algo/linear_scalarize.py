"""
Implementation of linear scalarization algorithm for Q-learning.

Reference
---------
    Fan, Z., Peng, N., Tian, M., & Fain, B. Welfare and Fairness in Multi-objective Reinforcement Learning.
    https://github.com/MuhangTian/Fair-MORL-AAMAS
"""
import numpy as np
from tqdm import tqdm
import wandb
from algo.utils import WelfareFunc

class LinearScalarize:
    def __init__(self, env, init_val, episodes, weights, lr, gamma, epsilon, welfare_func_name, save_path, nsw_lambda, p=None, seed=2023, wdb=False) -> None:
        self.env = env
        self.init_val = init_val
        self.welfare_func_name = "nash welfare" if welfare_func_name == "nash-welfare" else welfare_func_name
        print(f"LinearScalarize: {self.welfare_func_name}")
        self.episodes = episodes
        self.weights = weights
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.save_path = save_path
        self.seed = seed
        self.wdb = wdb
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p)
        self.dim = len(self.env.loc_coords)
        self.Racc_record = []
    
    def greedy(self, vec):
        '''Helper function'''
        arr = []
        for val in vec: 
            arr.append(np.dot(self.weights, val))
        return np.argmax(arr)
    
    def train(self):
        if len(self.weights) != self.dim: 
            raise ValueError('Dimension of weights not same as dimension of rewards')
        
        self.Q = np.full([self.env.observation_space.n, self.env.action_space.n, self.dim], self.init_val, dtype=float)
    
        for i in range(1, self.episodes+1):
            R_acc = np.zeros(self.dim)   # for recording performance, does not affect action selection
            state = self.env.reset()
            done = False
            c = 0
            
            while not done:
                if np.random.uniform(0,1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.greedy(self.Q[state])
                    
                next, reward, done = self.env.step(action)
                next_action = self.greedy(self.Q[next])
                
                for j in range(len(self.Q[state, action])):
                    self.Q[state,action][j] = self.Q[state,action][j] + self.lr*(reward[j]+self.gamma*self.Q[next,next_action][j]-self.Q[state,action][j])
                    
                state = next
                R_acc += np.power(self.gamma, c)*reward
                c += 1
            
            R_acc = np.where(R_acc < 0, 0, R_acc)
            
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
        np.savez(self.save_path, Q=self.Q, Racc_record=np.asarray(self.Racc_record)) 