"""
Implementation of Welfare Q-learning algorithm.

Reference
---------
    Fan, Z., Peng, N., Tian, M., & Fain, B. Welfare and Fairness in Multi-objective Reinforcement Learning.
    https://github.com/MuhangTian/Fair-MORL-AAMAS
"""
import numpy as np
from tqdm import tqdm
import wandb
from algo.utils import WelfareFunc


class WelfareQ:
    def __init__(self, env, lr, gamma, epsilon, episodes, init_val, welfare_func_name, nsw_lambda, save_path, dim_factor, p=None, non_stationary=True, seed=1122, wdb=False):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p)
        self.welfare_func_name = welfare_func_name
        self.p = p
        self.save_path = save_path
        self.seed = seed
        self.epsilon = epsilon
        self.wdb = wdb
        self.episodes = episodes
        self.init_val = init_val
        self.non_stationary = non_stationary
        self.nsw_lambda = nsw_lambda
        self.dim_factor = dim_factor
    
    def initialize(self):
        self.Q = np.zeros([self.env.observation_space.n, self.env.action_space.n, len(self.env.loc_coords)], dtype=float)
        self.Q = self.Q + self.init_val
        self.Racc_record = []
    
    def argmax_nsw(self, R, gamma_Q):
        '''Helper function for run_NSW_Q_learning'''
        sum = R + gamma_Q
        nsw_vals = [self.welfare_func(sum[i]) for i in range(self.env.action_space.n)]
        if np.all(nsw_vals == nsw_vals[0]) == True: # if all values are same, random action
            # numpy argmax always return first element when all elements are same
            action = self.env.action_space.sample()
        else:
            action = np.argmax(nsw_vals)
        return action
    
    def argmax_p_welfare(self, vec):
        '''Helper for Q welfare with input value of p'''
        arr = []
        for val in vec:
            sum = np.sum(np.power(val, self.p))
            arr.append(np.power(sum / len(val), 1/self.p))
        return np.argmax(arr)
    
    def argmax_egalitarian(self, R_acc, vec):
        '''Helepr function for egalitarian welfare'''
        idx = np.argmin(R_acc)
        arr = []
        for val in vec: 
            arr.append(val[idx])
        return np.argmax(arr)
        
    def train(self):
        self.initialize()
        
        for i in range(1, self.episodes+1):
            R_acc = np.zeros(len(self.env.loc_coords))
            state = self.env.reset()
            print('Episode {}\nInitial State: {}'.format(i, self.env.decode(state)))
            done = False
            # old_table = np.copy(Q)
            # avg = []
            c = 0
        
            while not done:
                
                if np.random.uniform(0,1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    if self.non_stationary:
                        if self.welfare_func_name == "p-welfare":
                            action = self.argmax_p_welfare(R_acc+np.power(self.gamma,c)*self.Q[state])
                        elif self.welfare_func_name == "nsw":
                            action = self.argmax_nsw(R_acc, np.power(self.gamma,c)*self.Q[state])
                        elif self.welfare_func_name == "egalitarian":
                            action = self.argmax_egalitarian(R_acc, R_acc+np.power(self.gamma,c)*self.Q[state])
                    else:   # if stationary policy, then Racc doesn't affect action selection
                        # action = self.argmax_nsw(0, self.Q[state])
                        raise NotImplementedError("Stationary policy is not implemented yet")
                        
                next, reward, done = self.env.step(action)
                
                if self.welfare_func_name == "p-welfare":
                    max_action = self.argmax_p_welfare(self.gamma*self.Q[next])
                elif self.welfare_func_name == "nsw":
                    max_action = self.argmax_nsw(0, self.gamma*self.Q[next])
                elif self.welfare_func_name == "egalitarian":
                    max_action = self.argmax_egalitarian(R_acc, self.gamma*self.Q[next])
                    
                self.Q[state, action] = self.Q[state, action] + self.lr*(reward + self.gamma*self.Q[next, max_action] - self.Q[state, action])
                
                self.epsilon = max(0.1, self.epsilon - self.dim_factor)  # epsilon diminish over time
                state = next
                R_acc += np.power(self.gamma,c)*reward
                c += 1
        
            R_acc = np.where(R_acc < 0, 0, R_acc) # Replace the negatives with 0
            
            if self.welfare_func_name == "nsw":
                nonlinear_score = np.power(np.product(R_acc), 1/len(R_acc))
            elif self.welfare_func_name in ["p-welfare", "egalitarian"]:
                nonlinear_score = self.welfare_func(R_acc)
                
            self.Racc_record.append(R_acc)
            print(f"R_acc: {R_acc}, {self.welfare_func_name}: {nonlinear_score}")
            
            if self.wdb:
                wandb.log({self.welfare_func_name: nonlinear_score})
        
        print("Finish training")
        np.savez(self.save_path, Q=self.Q, Racc_record=np.asarray(self.Racc_record))

