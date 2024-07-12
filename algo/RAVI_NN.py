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
    def __init__(
            self, 
            env, 
            gamma, 
            reward_dim, 
            time_horizon, 
            welfare_func_name, 
            nsw_lambda, 
            save_path, 
            seed=1122, 
            p=None, 
            threshold=5, 
            wdb=False, 
            scaling_factor=1, 
            hidden_dim=64, 
            lr=1e-3, 
            batch_size=128, 
            n_samples_per_timestep=10000, 
            grad_norm=1,
            avg_loss_interval=50,
        ) -> None:

        self.env = env
        self.grad_norm = grad_norm
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
        self.batch_size = batch_size
        self.n_samples_per_timestep = n_samples_per_timestep

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

        self.loss_record = []
        self.avg_loss_interval = avg_loss_interval

    def initialize(self):
        num_samples = 10000  # Define a fixed number of samples for initialization
        for _ in tqdm(range(num_samples), desc="Initializing Q-network..."):
            state = np.random.randint(self.env.observation_space.n)
            Racc = np.random.rand(self.reward_dim) * self.time_horizon / self.scaling_factor  # Randomly sample accumulated reward
            t = torch.tensor([0], dtype=torch.float32).unsqueeze(0).to(self.device)
            state_tensor = torch.tensor([state], dtype=torch.float32).unsqueeze(0).to(self.device)
            Racc_tensor = torch.tensor(Racc, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            output = self.q_network(state_tensor, Racc_tensor, t)
            # Compute the target value and replicate it for all actions
            target_value = self.welfare_func(Racc)
            target = torch.tensor([target_value] * self.action_dim, dtype=torch.float32).unsqueeze(0).to(self.device)
        
            self.q_network.zero_grad()
            loss = self.criterion(output, target)
            self.__record_loss(loss, interval=1000)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_norm)
            self.optimizer.step()

        self.loss_record = []       # clean loss record after initialization

    def train(self):
        self.initialize()

        for t in tqdm(range(1, self.time_horizon + 1), desc="Training..."):
            self.replay_buffer = []

            for _ in range(self.n_samples_per_timestep):
                state = np.random.randint(self.env.observation_space.n)
                Racc = np.random.rand(self.reward_dim) * (self.time_horizon - t) / self.scaling_factor  # Randomly sample accumulated reward
                transition_prob, reward_arr, next_state_arr = self.env.get_transition(state)
                self.replay_buffer.append((state, Racc, t, transition_prob, reward_arr, next_state_arr))

            self.train_one_epoch()      # train for one epoch after all the samples are collected

        self.test_evaluate(final=True)
    
    def train_one_epoch(self):
        indices = np.arange(len(self.replay_buffer))
        np.random.shuffle(indices)
        iters = len(self.replay_buffer) // self.batch_size
        for i in range(iters):
            batch_samples = self.replay_buffer[i * self.batch_size: (i + 1) * self.batch_size]
            self.update_q_network(batch_samples)

    def update_q_network(self, batch):
        # Unpack the batch into separate lists
        states, Raccs, ts, transition_probs, reward_arrs, next_state_arrs = zip(*batch)

        # Convert lists to tensors and move to the appropriate device
        states = torch.tensor(states, dtype=torch.float32).to(self.device) # (bsz)
        Raccs = torch.tensor(Raccs, dtype=torch.float32).to(self.device).squeeze(-1) # (bsz)
        ts = torch.tensor(ts, dtype=torch.float32).to(self.device) # (bsz)
        transition_probs = torch.tensor(transition_probs, dtype=torch.float32).to(self.device)  # (bsz, n_actions)
        reward_arrs = torch.tensor(reward_arrs, dtype=torch.float32).to(self.device) # (bsz, n_actions, 1)
        next_state_arrs = torch.tensor(next_state_arrs, dtype=torch.long).to(self.device)  # (bsz, n_actions)

        # Initialize a tensor to store the target Q-values for each action
        target_q_values = torch.zeros((len(batch), self.num_actions)).to(self.device)   # (bsz, n_actions)

        # Compute the target Q-values for each action
        with torch.no_grad():
            for i in range(len(batch)):
                for a in range(self.num_actions):
                    next_state = next_state_arrs[i, a]  # Make it a 1D tensor
                    next_Racc = Raccs[i] + torch.pow(self.gamma, self.time_horizon - ts[i]) * reward_arrs[i, a]   # 1d
                    next_q_values = self.q_network(next_state.unsqueeze(0), next_Racc, (ts[i] - 1).unsqueeze(0))
                    target_q_values[i, a] = torch.max(next_q_values * transition_probs[i, a])

                    # Print statements for debugging
                    # print(f"next_state: {next_state}")
                    # print(f"next_Racc: {next_Racc}")
                    # print(f"next_q_values: {next_q_values}")

                    # target_q_values[i, a] = reward_arrs[i, a] + self.gamma * max_next_q_value * transition_probs[i, a]

        # Get the Q-values for the current state-action pairs
        # print(state)
        # print(states.shape)
        # print(Raccs)
        # print(Raccs.shape)
        # print(ts)
        # print(ts.shape)
        # q_values = self.q_network(states.repeat(Raccs.shape[1], 1).transpose(1, 0).unsqueeze(-1), Raccs, ts)
        q_values = self.q_network(states.unsqueeze(-1), Raccs.unsqueeze(-1), ts.unsqueeze(-1))

        # print(f"q_values: {q_values}")
        # print(q_values.shape)
        # print(f"target_q_values: {target_q_values}")
        
        # Compute the loss between the current Q-values and the target Q-values
        loss = self.criterion(q_values, target_q_values)
        self.__record_loss(loss)

        # Zero out the gradients, backpropagate the loss, and update the network parameters
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_norm)
        self.optimizer.step()
    
    def __record_loss(self, loss, interval=None):      # private method
        loss_interval = interval if interval is not None else self.avg_loss_interval

        if len(self.loss_record) == loss_interval:
            # assert len(self.loss_record) == self.avg_loss_interval, "Number of losses is wrong!"
            print(f"Average loss for every {loss_interval} iterations: {np.mean(self.loss_record)}")
            self.loss_record = []
        else:
            assert len(self.loss_record) < loss_interval, f"Loss record should not exceed {loss_interval}"
            self.loss_record.append(loss.item())

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

            # Detailed printout for each timestep
            print(f"Timestep {self.time_horizon - t + 1}")
            print(f"State: {state}")
            print(f"Q-values: {q_values.cpu().detach().numpy()}")
            print(f"Action taken: {action}")

            next, reward, done = self.env.step(action)
            if isinstance(self.env, envs.Resource_Gathering.ResourceGatheringEnv):
                img_path = self.save_path + f'/renders/env_render_{self.time_horizon-t}.png'
                self.env.render(save_path=img_path)
            state = next
            Racc += np.power(self.gamma, c) * reward
            print(f"Accumulated Reward at t = {self.time_horizon-t}: {Racc}\n")

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

    def test_evaluate(self, final=False):
        initial_state = self.env.reset(seed=self.seed)
        renders_path = self.save_path + '/renders'
        os.makedirs(renders_path, exist_ok=True)
        img_path = self.save_path + f'/renders/env_render_init.png'
        if isinstance(self.env, envs.Resource_Gathering.ResourceGatheringEnv):
            self.env.render(save_path=img_path)
        Racc = np.zeros(self.reward_dim)
        c = 0

        for t in range(self.time_horizon, 0, -1):
            print(f"\n=== Timestep {self.time_horizon - t + 1} ===")

            for state in range(self.env.observation_space.n):
                state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
                Racc_tensor = torch.tensor(Racc, dtype=torch.float32).to(self.device)
                t_tensor = torch.tensor([t], dtype=torch.float32).to(self.device)
                
                # Find the action that maximizes the Q-value
                q_values = self.q_network(state_tensor, Racc_tensor, t_tensor)
                action = torch.argmax(q_values).item()

                # Decode the state to get the taxi location and passenger information
                taxi_x, taxi_y, pass_idx = self.env.decode(state)
                taxi_loc = (taxi_x, taxi_y)
                has_passenger = pass_idx is not None and pass_idx != len(self.env.dest_coords)
                pass_dest = self.env.dest_coords[pass_idx] if has_passenger else None

                # Detailed printout for each timestep
                print(f"State: {state}")
                print(f"Taxi location: {taxi_loc}, Has passenger: {has_passenger}, Passenger destination: {pass_dest}")
                print(f"Q-values: {q_values.cpu().detach().numpy()}")
                print(f"Optimal action: {action}")
            
            # Perform the action for the initial state to proceed in the environment
            initial_state_tensor = torch.tensor([initial_state], dtype=torch.float32).to(self.device)
            initial_Racc_tensor = torch.tensor(Racc, dtype=torch.float32).to(self.device)
            initial_t_tensor = torch.tensor([t], dtype=torch.float32).to(self.device)
            q_values = self.q_network(initial_state_tensor, initial_Racc_tensor, initial_t_tensor)
            action = torch.argmax(q_values).item()

            next_state, reward, done = self.env.step(action)
            if isinstance(self.env, envs.Resource_Gathering.ResourceGatheringEnv):
                img_path = self.save_path + f'/renders/env_render_{self.time_horizon-t}.png'
                self.env.render(save_path=img_path)
            
            initial_state = next_state
            Racc += np.power(self.gamma, c) * reward
            print(f"Accumulated Reward at t = {self.time_horizon-t}: {Racc}\n")

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