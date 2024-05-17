import gym
from gym import spaces
import numpy as np

class ResourceGatheringEnv(gym.Env):
    def __init__(self, grid_size=(5, 5), num_resources=3):
        super(ResourceGatheringEnv, self).__init__()
        self.grid_size = grid_size
        self.num_resources = num_resources

        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Total number of possible states
        self.total_states = (grid_size[0] * grid_size[1]) * (2 ** num_resources)
        
        # Observation space: Discrete space of all possible states
        self.observation_space = spaces.Discrete(self.total_states)

        # Fixed positions for resources and enemies
        self.resource_positions = [(1, 1), (3, 3), (4, 0)]
        self.enemy_positions = [(0, 4), (4, 4)]

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Initialize agent position
        self.agent_pos = [self.np_random.randint(self.grid_size[0]), self.np_random.randint(self.grid_size[1])]
        
        # Initialize resource status (1 = present, 0 = collected)
        self.resources_status = np.ones(self.num_resources, dtype=np.int32)

        self.collected_resources = 0
        self.damage_taken = 0

        return self._get_obs()

    def _get_obs(self):
        # Flatten agent position
        agent_position_flat = self.agent_pos[0] * self.grid_size[1] + self.agent_pos[1]

        # Convert resource status to a single integer (binary representation)
        resource_status_int = int(''.join(map(str, self.resources_status)), 2)

        # Combine agent position and resource status into a single state
        obs = agent_position_flat * (2 ** self.num_resources) + resource_status_int

        return obs

    def step(self, action):
        if action == 0 and self.agent_pos[0] > 0: # left
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size[0] - 1: # right
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0: # down
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size[1] - 1: # up
            self.agent_pos[1] += 1

        reward = self.compute_reward()
        done = False  # Continuing task: never done
        return self._get_obs(), reward, done

    def compute_reward(self):
        agent_pos_tuple = tuple(self.agent_pos)

        # Check if agent is on a resource
        if agent_pos_tuple in self.resource_positions:
            index = self.resource_positions.index(agent_pos_tuple)
            if self.resources_status[index] == 1:
                self.collected_resources += 1
                self.resources_status[index] = 0

        # Check if agent is on an enemy
        if agent_pos_tuple in self.enemy_positions:
            self.damage_taken += 1

        return np.array([self.collected_resources, self.damage_taken])

    def render(self, mode='human'):
        print(f"Agent Position: {self.agent_pos}, Resources Collected: {self.collected_resources}, Damage Taken: {self.damage_taken}")

    def decode_state(self, state):
        # Extract resource status
        resource_status_int = state % (2 ** self.num_resources)
        # Extract agent position
        agent_position_flat = state // (2 ** self.num_resources)

        # Convert agent position to 2D coordinates
        x = agent_position_flat // self.grid_size[1]
        y = agent_position_flat % self.grid_size[1]
        agent_pos = [x, y]

        # Convert resource status integer to binary vector
        resource_status_bin = format(resource_status_int, f'0{self.num_resources}b')
        resource_status = np.array(list(map(int, resource_status_bin)), dtype=np.int32)

        return agent_pos, resource_status
    
    def get_transition(self, state: int):
        """
        Given a state (s), return all the transition probabilities Pr(s'|s,a) and R(s,a) for all possible actions.

        Parameters
        ----------
        state : int
            integer encoding of the state
            
        Returns
        -------
        transition_prob : array
            transition probability for each action
        reward_arr : array
            reward for each action
        next_state_arr : array
            next state for each action
        """
        # Decode the state
        agent_pos, resource_status = self.decode_state(state)
        transition_prob = np.zeros(self.action_space.n)
        reward_arr = np.zeros((self.action_space.n, 2))  # Reward is now a vector [resources_collected, damage_taken]
        next_state_arr = np.zeros(self.action_space.n, dtype=int)

        # For each action, calculate transition probability, reward, and next state
        for action in range(self.action_space.n):
            # Copy current state
            new_agent_pos = agent_pos.copy()
            new_resource_status = resource_status.copy()

            # Determine new position based on action
            if action == 0 and new_agent_pos[0] > 0:  # left
                new_agent_pos[0] -= 1
            elif action == 1 and new_agent_pos[0] < self.grid_size[0] - 1:  # right
                new_agent_pos[0] += 1
            elif action == 2 and new_agent_pos[1] > 0:  # down
                new_agent_pos[1] -= 1
            elif action == 3 and new_agent_pos[1] < self.grid_size[1] - 1:  # up
                new_agent_pos[1] += 1

            # Calculate reward
            reward = np.array([0, 0])  # Initialize reward as [0, 0]
            new_pos_tuple = tuple(new_agent_pos)
            if new_pos_tuple in self.resource_positions:
                index = self.resource_positions.index(new_pos_tuple)
                if new_resource_status[index] == 1:
                    reward[0] = 1  # Collecting a resource
                    new_resource_status[index] = 0  # Resource is now collected
            if new_pos_tuple in self.enemy_positions:
                reward[1] = 1  # Encountering an enemy

            # Encode new state
            new_agent_flat = new_agent_pos[0] * self.grid_size[1] + new_agent_pos[1]
            new_resource_status_int = int(''.join(map(str, new_resource_status)), 2)
            next_state = new_agent_flat * (2 ** self.num_resources) + new_resource_status_int

            # Update arrays
            transition_prob[action] = 1  # Deterministic transitions
            reward_arr[action] = reward
            next_state_arr[action] = next_state

        return transition_prob, reward_arr, next_state_arr

if __name__ == "__main__":
    # To use the environment and test decoding
    env = ResourceGatheringEnv(grid_size=(5, 5), num_resources=3)
    state = env.reset(seed=42)  # Reset with a specific seed
    for _ in range(10):  # Simulate a few steps
        action = env.action_space.sample()
        state, reward, done = env.step(action)
        env.render()
        decoded_pos, decoded_status = env.decode_state(state)
        print(f"Decoded Position: {decoded_pos}, Decoded Resource Status: {decoded_status}")

        # Test get_transition function
        transition_prob, reward_arr, next_state_arr = env.get_transition(state)
        print(f"Transition Probabilities: {transition_prob}")
        print(f"Rewards: {reward_arr}")
        print(f"Next States: {next_state_arr}")
    env.close()