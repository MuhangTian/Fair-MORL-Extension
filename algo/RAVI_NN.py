import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import numpy as np
from algo.utils import WelfareFunc
import wandb
import sys
import envs
import os

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(1 + 1 + reward_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output a single Q value

    def forward(self, state, action, Racc, t):
        x = torch.cat((state, action, Racc, t), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
    
class RAVI_NN:
    def __init__(self, env, gamma, reward_dim, time_horizon, welfare_func_name, nsw_lambda, save_path, seed=1122, p=None, threshold=5, wdb=False, scaling_factor=1, hidden_dim=64, lr=1e-3) -> None:
        self.env = env
        self.welfare_func_name = "nash welfare" if welfare_func_name == "nash-welfare" else welfare_func_name
        self.gamma = gamma
        self.scaling_factor = scaling_factor
        self.reward_dim = reward_dim
        self.time_horizon = time_horizon
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p, threshold)
        self.training_complete = False
        self.seed = seed
        self.wdb = wdb
        self.save_path = save_path

        self.num_actions = env.action_space.n  # Get the number of actions from the environment
        # Define Q-network and optimizer
        self.state_dim = self.env.observation_space.n
        self.action_dim = self.num_actions
        self.hidden_dim = hidden_dim
        self.q_network = QNetwork(self.state_dim, self.action_dim, self.reward_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Buffer to store (state, action, Racc_code, t, reward, next_state, next_Racc)
        self.replay_buffer = []

    def initialize(self):
        num_samples = 1000  # Define a fixed number of samples for initialization
        for _ in tqdm(range(num_samples), desc="Initializing Q-network..."):
            state = np.random.randint(self.env.observation_space.n)
            action = np.random.randint(self.num_actions)
            Racc = np.random.rand(self.reward_dim) * self.time_horizon / self.scaling_factor  # Randomly sample accumulated reward
            t = torch.tensor([0], dtype=torch.float32).unsqueeze(0)
            state_tensor = torch.tensor([state], dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0)
            Racc_tensor = torch.tensor(Racc, dtype=torch.float32).unsqueeze(0)
            
            output = self.q_network(state_tensor, action_tensor, Racc_tensor, t)
            target = torch.tensor(self.welfare_func(Racc), dtype=torch.float32).unsqueeze(0)
        
            self.q_network.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def train(self):
        self.initialize()
        
        num_samples = 1000  # Define a fixed number of samples per iteration
        
        for t in tqdm(range(1, self.time_horizon + 1), desc="Training..."):
            for _ in range(num_samples):
                state = np.random.randint(self.env.observation_space.n)
                Racc = np.random.rand(self.reward_dim) * (self.time_horizon - t) / self.scaling_factor  # Randomly sample accumulated reward
                transition_prob, reward_arr, next_state_arr = self.env.get_transition(state)

                next_Racc = Racc + np.power(self.gamma, self.time_horizon - t) * reward_arr
                # print(next_Racc)
                # print()

                for a in range(self.num_actions):
                    self.replay_buffer.append((state, a, Racc, t, reward_arr[a], next_state_arr[a], next_Racc[a], transition_prob[a]))

                if len(self.replay_buffer) > 32:
                    batch = random.sample(self.replay_buffer, 32)
                    self.update_q_network(batch, t)

        self.evaluate(final=True)

    def update_q_network(self, batch, t):
        states, actions, Raccs, ts, rewards, next_states, next_Raccs, transition_probs = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).unsqueeze(-1)
        actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1)
        Raccs = torch.tensor(Raccs, dtype=torch.float32)
        ts = torch.tensor(ts, dtype=torch.float32).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32).unsqueeze(-1)
        next_Raccs = torch.tensor(next_Raccs, dtype=torch.float32)
        transition_probs = torch.tensor(transition_probs, dtype=torch.float32)

        with torch.no_grad():
            target_q_values = []
            for i in range(len(batch)):
                next_q_values = []
                for a in range(self.num_actions):
                    # print(next_Raccs[i])
                    next_q_value = self.q_network(next_states[i], torch.tensor([a], dtype=torch.float32), next_Raccs[i], ts[i] - 1)
                    next_q_values.append(next_q_value)
                target_q_value = rewards[i] + self.gamma * max(next_q_values) * transition_probs[i]
                target_q_values.append(target_q_value)
            target_q_values = torch.stack(target_q_values)

        q_values = self.q_network(states, actions, Raccs, ts)
        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self, final=False):
        state = self.env.reset(seed=self.seed)
        renders_path = self.save_path + '/renders'
        os.makedirs(renders_path, exist_ok=True)
        img_path = self.save_path + f'/renders/env_render_init.png'
        if isinstance(self.env, envs.Resource_Gathering.ResourceGatheringEnv):
            self.env.render(save_path=img_path)
        Racc = np.zeros(self.reward_dim)
        c = 0

        for t in range(self.time_horizon, 0, -1):
            state_tensor = torch.tensor([state], dtype=torch.float32).unsqueeze(0)
            Racc_tensor = torch.tensor(Racc, dtype=torch.float32).unsqueeze(0)
            t_tensor = torch.tensor([t], dtype=torch.float32).unsqueeze(0)
            
            action = torch.argmax(self.q_network(state_tensor, input_tensor, Racc_tensor, t_tensor)).item()

            next, reward, done = self.env.step(action)
            if isinstance(self.env, envs.Resource_Gathering.ResourceGatheringEnv):
                img_path = self.save_path + f'/renders/env_render_{self.time_horizon-t}.png'
                self.env.render(save_path=img_path)
            state = next
            Racc += np.power(self.gamma, c) * reward
            print(f"Accumulated Reward at t = {self.time_horizon-t}: {Racc}")

            c += 1

        if self.welfare_func_name == "nash welfare":
            if self.wdb:
                wandb.log({self.welfare_func_name: self.welfare_func.nash_welfare(Racc)})
            print(f"{self.welfare_func_name}: {self.welfare_func.nash_welfare(Racc)}, Racc: {Racc}")
        elif self.welfare_func_name in ["p-welfare", "egalitarian", "RD-threshold", "Cobb-Douglas"]:
            if self.wdb:
                wandb.log({self.welfare_func_name: self.welfare_func(Racc)})
            print(f"{self.welfare_func_name}: {self.welfare_func(Racc)}, Racc: {Racc}")

        if final:
            self.Racc_record = Racc