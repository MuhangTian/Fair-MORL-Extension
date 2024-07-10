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
    def __init__(self, state_dim, action_dim, reward_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(1 + reward_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # Output Q value for each action

    def forward(self, state, Racc, t):
        x = torch.cat((state, Racc, t), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
    
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

        # Set device to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

        self.num_actions = env.action_space.n  # Get the number of actions from the environment
        # Define Q-network and optimizer
        self.state_dim = self.env.observation_space.n
        self.action_dim = self.num_actions
        self.hidden_dim = hidden_dim
        self.q_network = QNetwork(self.state_dim, self.action_dim, self.reward_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Buffer to store (state, action, Racc, t, reward, next_state, next_Racc, transition_prob)
        self.replay_buffer = []

    def initialize(self):
        num_samples = 1000  # Define a fixed number of samples for initialization
        for _ in tqdm(range(num_samples), desc="Initializing Q-network..."):
            state = np.random.randint(self.env.observation_space.n)
            Racc = np.random.rand(self.reward_dim) * self.time_horizon / self.scaling_factor  # Randomly sample accumulated reward
            t = torch.tensor([0], dtype=torch.float32).unsqueeze(0).to(self.device)
            state_tensor = torch.tensor([state], dtype=torch.float32).unsqueeze(0).to(self.device)
            Racc_tensor = torch.tensor(Racc, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            output = self.q_network(state_tensor, Racc_tensor, t)
            target = torch.tensor(self.welfare_func(Racc), dtype=torch.float32).unsqueeze(0).to(self.device)
        
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

                Racc_replicated = np.tile(Racc, (self.num_actions, 1))
                t_replicated = np.tile(t, (self.num_actions, 1))

                # Store the experience
                self.replay_buffer.append((state, Racc_replicated, t_replicated, transition_prob, reward_arr, next_state_arr))

                if len(self.replay_buffer) > 32:
                    batch = random.sample(self.replay_buffer, 32)
                    self.update_q_network(batch)

        self.evaluate(final=True)

    def update_q_network(self, batch):
        # Unpack the batch into separate lists
        states, Raccs, ts, transition_probs, reward_arrs, next_state_arrs = zip(*batch)

        # Convert lists to tensors and move to the appropriate device
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        Raccs = torch.tensor(Raccs, dtype=torch.float32).to(self.device)
        ts = torch.tensor(ts, dtype=torch.float32).to(self.device)
        transition_probs = torch.tensor(transition_probs, dtype=torch.float32).to(self.device)
        reward_arrs = torch.tensor(reward_arrs, dtype=torch.float32).to(self.device)
        next_state_arrs = torch.tensor(next_state_arrs, dtype=torch.long).to(self.device)  # Ensure long type for indexing

        # Initialize a tensor to store the target Q-values for each action
        target_q_values = torch.zeros((len(batch), self.num_actions)).to(self.device)

        # Compute the target Q-values for each action
        with torch.no_grad():
            for i in range(len(batch)):
                for a in range(self.num_actions):
                    next_state = next_state_arrs[i, a].unsqueeze(0)  # Make it a 1D tensor
                    next_Racc = Raccs[i, a] + torch.pow(self.gamma, self.time_horizon - ts[i, a]) * reward_arrs[i, a]
                    next_q_values = self.q_network(next_state.unsqueeze(0), next_Racc.unsqueeze(0), (ts[i, a] - 1).unsqueeze(0))
                    target_q_values[i, a] = torch.max(next_q_values * transition_probs[i, a])

                    # Print statements for debugging
                    # print(f"next_state: {next_state}")
                    # print(f"next_Racc: {next_Racc}")
                    # print(f"next_q_values: {next_q_values}")

                    # target_q_values[i, a] = reward_arrs[i, a] + self.gamma * max_next_q_value * transition_probs[i, a]

        # Get the Q-values for the current state-action pairs
        print(state)
        print(states.shape)
        print(Raccs)
        print(Raccs.shape)
        print(ts)
        print(ts.shape)
        q_values = self.q_network(states, Raccs, ts)

        print(f"q_values: {q_values}")
        print(q_values.shape)
        print(f"target_q_values: {target_q_values}")
        
        # Compute the loss between the current Q-values and the target Q-values
        loss = self.criterion(q_values, target_q_values)

        # Zero out the gradients, backpropagate the loss, and update the network parameters
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
            state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
            Racc_tensor = torch.tensor(Racc, dtype=torch.float32).to(self.device)
            t_tensor = torch.tensor([t], dtype=torch.float32).to(self.device)
            
            # Find the action that maximizes the Q-value
            q_values = self.q_network(state_tensor, Racc_tensor, t_tensor)
            action = torch.argmax(q_values).item()

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