import itertools

import numpy as np
from tqdm import tqdm
import sys
from sys import getsizeof

import wandb
from algo.utils import DiscreFunc, WelfareFunc

sys.path.insert(0, '../envs')
import envs

import os
import pdb
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class RAValueIteration:
    def __init__(self, env, discre_alpha, growth_rate, gamma, reward_dim, time_horizon, welfare_func_name, nsw_lambda, save_path, seed=1122, p=None, threshold=5, wdb=False, scaling_factor=1) -> None:
        self.env = env
        self.welfare_func_name = "nash welfare" if welfare_func_name == "nash-welfare" else welfare_func_name
        self.discre_alpha = discre_alpha # Discretization factor for accumulated rewards.
        self.discre_func = DiscreFunc(discre_alpha, growth_rate)  # Discretization function with growth rate for exponential discretization.
        self.gamma = gamma
        self.growth_rate = growth_rate
        self.scaling_factor = scaling_factor
        self.reward_dim = reward_dim
        self.time_horizon = time_horizon
        self.welfare_func = WelfareFunc(welfare_func_name, nsw_lambda, p, threshold)
        self.training_complete = False
        self.seed = seed
        self.wdb = wdb
        self.save_path = save_path

        self.num_actions = env.action_space.n  # Get the number of actions from the environment

        # Define CUDA kernel
        self.mod = SourceModule(f"""
#include <stdio.h>
#include <math.h>

__device__ float atomicMaxFloat(float* address, float val) {{
    int* address_as_int = (int*) address;
    int old = *address_as_int, assumed;

    do {{
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    }} while (assumed != old);

    return __int_as_float(old);
}}
                                
__device__ int atomicMaxInt(int* address, int val) {{
    int old = *address, assumed;
    do {{
        assumed = old;
        old = atomicCAS(address, assumed, max(val, assumed));
    }} while (assumed != old);
    return old;
}}

__device__ int discretize(float value, float alpha, float growth_rate) {{
    if (growth_rate > 1.0f) {{
        int index = roundf(log1pf(value / alpha * (growth_rate - 1.0f)) / logf(growth_rate));
        if (index < 0) {{
            // printf("Error: index < 0, value: %f, alpha: %f, growth_rate: %f, index: %d\\n", value, alpha, growth_rate, index);
            return 0;
        }}
        return index;
    }} else {{
        return roundf(value / alpha);
    }}
}}

__device__ int calculate_Racc_code(float *discre_grid, int grid_size, float *next_Racc, int reward_dim) {{
    int Racc_code = 0;
    int factor = 1;
    for (int i = reward_dim - 1; i >= 0; i--) {{
        int idx = -1;
        float min_diff = 1e6;
        for (int j = 0; j < grid_size; j++) {{
            float diff = fabs(next_Racc[i] - discre_grid[j]);
            // printf("Dim %d: next_Racc[i] = %f, discre_grid[j] = %f, diff = %f\\n", i, next_Racc[i], discre_grid[j], diff);
            if (diff < min_diff) {{
                min_diff = diff;
                idx = j;
            }}
        }}
        // printf("Dim %d: next_Racc[i] = %f, closest discre_grid value = %f, idx = %d\\n", i, next_Racc[i], discre_grid[idx], idx);
        Racc_code += idx * factor;
        factor *= grid_size;
    }}
    // printf("Computed Racc_code: %d\\n", Racc_code);
    return Racc_code;
}}

extern "C" __global__ void compute_values(float *V, int *Pi, float *transition_prob, float *reward_arr, int *next_state_arr,
                        float gamma, int time_horizon, float *curr_Racc, int *Racc_code, 
                        int num_states, int num_Racc_total, int num_Racc, int reward_dim, int num_actions, int t, float alpha, float growth_rate, float *discre_grid, int grid_size) {{
    extern __shared__ float shared_memory[];
    float *next_Racc = shared_memory;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states * num_Racc) return;

    int Racc_idx = idx / num_states;
    int state_idx = idx % num_states;

    if (Racc_idx >= num_Racc || state_idx >= num_states) {{
        // printf("Error: Racc_idx (%d) or state_idx (%d) out of bounds (num_Racc=%d, num_states=%d)\\n", Racc_idx, state_idx, num_Racc, num_states);
        return;
    }}

    int state = state_idx;
    int Racc_code_idx = Racc_code[Racc_idx];

    int V_idx = state * num_Racc_total * (time_horizon + 1) + Racc_code_idx * (time_horizon + 1) + t;

    // Ensure V_idx is within bounds
    if (V_idx >= num_states * num_Racc_total * (time_horizon + 1)) {{
        // printf("Error: V_idx out of bounds. V_idx=%d, Max=%d\\n", V_idx, num_states * num_Racc_total * (time_horizon + 1));
        return;
    }}

    float max_V = -INFINITY;
    int best_action = 0;

    // if (state == 5) {{
    //     printf("State: %d, Racc: [", state);
    //     for (int i = 0; i < reward_dim; i++) {{
    //         printf("%f", curr_Racc[Racc_idx * reward_dim + i]);
    //         if (i < reward_dim - 1) {{
    //             printf(", ");
    //         }}
    //     }}
    //     printf("], Racc_code: %d\\n", Racc_code_idx);

        for (int a = 0; a < num_actions; a++) {{
            float transition_probability = transition_prob[state * num_actions + a];
            int next_state = next_state_arr[state * num_actions + a];

            for (int r = 0; r < reward_dim; r++) {{
                next_Racc[threadIdx.x * reward_dim + r] = curr_Racc[Racc_idx * reward_dim + r] + powf(gamma, time_horizon - t) * reward_arr[(state * num_actions + a) * reward_dim + r];
            }}

            // printf("  action: %d, transition_probability: %f, next_state: %d\\n", a, transition_probability, next_state);
            // printf("  Next_Racc: [");
            // for (int r = 0; r < reward_dim; r++) {{
            //     printf("%f", next_Racc[threadIdx.x * reward_dim + r]);
            //     if (r < reward_dim - 1) {{
            //         printf(", ");
            //     }}
            // }}
            // printf("]\\n");

            int next_Racc_code = calculate_Racc_code(discre_grid, grid_size, next_Racc + threadIdx.x * reward_dim, reward_dim);

            // printf("  Next_Racc_code: %d\\n", next_Racc_code);

            float all_V = 0.0;
            
            int next_idx = next_state * num_Racc_total * (time_horizon + 1) + next_Racc_code * (time_horizon + 1) + (t - 1);
            // Ensure next_idx is within bounds
            if (next_idx >= num_states * num_Racc_total * (time_horizon + 1)) {{
                // printf("Error: next_idx out of bounds. next_idx=%d, Max=%d\\n", next_idx, num_states * num_Racc_total * (time_horizon + 1));
                return;
            }}
            all_V += transition_probability * V[next_idx];
            // printf("  Accessing V[%d, %d, %d] (flattened idx: %d) = %f\\n", next_state, next_Racc_code, t - 1, next_idx, V[next_idx]);

            if (all_V > max_V) {{
                max_V = all_V;
                best_action = a;
            }}
        }}

        // printf("  max_V: %f, best_action: %d\\n", max_V, best_action);
        // printf("  Updating V[%d] (flattened idx: %d) with %f\\n", V_idx, V_idx, max_V);
        // printf("  Updating Pi[%d] (flattened idx: %d) with %d\\n", V_idx, V_idx, best_action);
    // }}

    atomicMaxFloat(&V[V_idx], max_V);
    atomicMaxInt(&Pi[V_idx], best_action);
}}
""")
    
    def initialize(self):
        # min, max = 0, np.floor(self.time_horizon / self.discre_alpha) * self.discre_alpha
        # self.discre_grid = np.arange(min, max + self.discre_alpha, self.discre_alpha)

        # Calculate maximum possible reward accumulation for exponential discretization
        if self.gamma == 1:  # Special case where gamma is 1, just multiply max reward by the number of steps
            max_reward = self.time_horizon/self.scaling_factor
        else:
            # Calculate the sum of the geometric series
            sum_of_discounts = (1 - self.gamma ** ((self.time_horizon+1)/self.scaling_factor)) / (1 - self.gamma)
            max_reward = sum_of_discounts
        max_discrete = self.discre_func(max_reward)

        print(f"Max discretized reward during initialization: {max_discrete}")

        self.discre_grid = np.unique(np.array([self.discre_func(alpha * self.discre_alpha) for alpha in range(0, int(np.ceil(max_discrete / self.discre_alpha)) + 1)]))
        # for alpha in range(0, int(np.ceil(max_discrete / self.discre_alpha)) + 1):
        #     print(f'{alpha*self.discre_alpha}, {self.discre_func(alpha * self.discre_alpha)}')
        # print(self.discre_grid)
        self.init_Racc = [np.asarray(r) for r in list(itertools.product(self.discre_grid, repeat=self.reward_dim))] # All possible reward accumulations.
        self.encode_Racc_dict = {str(r): i for i, r in enumerate(self.init_Racc)} # Encoding of reward accumulations.

        # print("Encoding dictionary")
        # print(self.encode_Racc_dict)

        self.flat_encode_Racc_dict, self.flat_encode_Racc_str_lens = self.create_flat_encode_dict()

        # print("Flat encoding dictionary")
        # print(self.flat_encode_Racc_dict)
        
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
    
    def create_flat_encode_dict(self):
        max_index = max(self.encode_Racc_dict.values())
        flat_dict = np.full((max_index + 1, 256), -1, dtype=np.int32)  # Initialize with -1 to detect invalid accesses
        str_lens = np.zeros(max_index + 1, dtype=np.int32)  # Lengths of strings for each index

        for k, v in self.encode_Racc_dict.items():
            for i, char in enumerate(k):
                flat_dict[v, i] = ord(char)  # Convert char to ASCII int
            str_lens[v] = len(k)

        return flat_dict.flatten(), str_lens

    # Function to launch CUDA kernel
    def parallel_compute(self, V, Pi, transition_prob, reward_arr, next_state_arr, gamma, time_horizon, curr_Racc, Racc_code, num_states, num_Racc_total, num_Racc, reward_dim, num_actions, t, alpha, growth_rate, discre_grid, grid_size):
        V_flat = V.flatten().astype(np.float32)
        Pi_flat = Pi.flatten().astype(np.int32)  # Ensure Pi is treated as an integer array

        # print("V_flat")
        # print(V_flat)
        
        V_gpu = cuda.mem_alloc(V_flat.nbytes)
        Pi_gpu = cuda.mem_alloc(Pi_flat.nbytes)
        transition_prob_gpu = cuda.mem_alloc(transition_prob.nbytes)
        reward_arr_gpu = cuda.mem_alloc(reward_arr.nbytes)
        next_state_arr_gpu = cuda.mem_alloc(next_state_arr.nbytes)
        curr_Racc_gpu = cuda.mem_alloc(curr_Racc.nbytes)
        Racc_code_gpu = cuda.mem_alloc(Racc_code.nbytes)
        discre_grid_gpu = cuda.mem_alloc(discre_grid.nbytes)
        
        cuda.memcpy_htod(V_gpu, V_flat)
        cuda.memcpy_htod(Pi_gpu, Pi_flat)
        cuda.memcpy_htod(transition_prob_gpu, transition_prob)
        cuda.memcpy_htod(reward_arr_gpu, reward_arr)
        cuda.memcpy_htod(next_state_arr_gpu, next_state_arr)
        cuda.memcpy_htod(curr_Racc_gpu, curr_Racc)
        cuda.memcpy_htod(Racc_code_gpu, Racc_code)
        cuda.memcpy_htod(discre_grid_gpu, discre_grid)
        
        block_size = 256
        grid_size = (num_states * num_Racc + block_size - 1) // block_size
        shared_mem_size = block_size * (reward_dim * np.dtype(np.float32).itemsize)  # Size of shared memory per thread

        # print_discre_grid = self.mod.get_function("print_discre_grid")
        # print_discre_grid(discre_grid_gpu, np.int32(len(discre_grid)), block=(1, 1, 1), grid=(1, 1, 1))

        func = self.mod.get_function("compute_values")
        func(V_gpu, Pi_gpu, transition_prob_gpu, reward_arr_gpu, next_state_arr_gpu, np.float32(gamma), np.int32(time_horizon), curr_Racc_gpu, Racc_code_gpu, np.int32(num_states), np.int32(num_Racc_total), np.int32(num_Racc), np.int32(reward_dim), np.int32(num_actions), np.int32(t), np.float32(alpha), np.float32(growth_rate), discre_grid_gpu, np.int32(len(discre_grid)), block=(block_size, 1, 1), grid=(grid_size, 1, 1), shared=shared_mem_size)
        
        cuda.memcpy_dtoh(V_flat, V_gpu)
        cuda.memcpy_dtoh(Pi_flat, Pi_gpu)

        # print("V_flat")
        # print(V_flat)

        # print("V_flat.reshape(V.shape)")
        # print(V_flat.reshape(V.shape))

        # print("Pi_flat")
        # print(Pi_flat)

        # print("Pi_flat.reshape(Pi.shape)")
        # print(Pi_flat.reshape(Pi.shape))
        
        V[:] = V_flat.reshape(V.shape)
        Pi[:] = Pi_flat.reshape(Pi.shape)
        
        V_gpu.free()
        Pi_gpu.free()
        transition_prob_gpu.free()
        reward_arr_gpu.free()
        next_state_arr_gpu.free()
        curr_Racc_gpu.free()
        Racc_code_gpu.free()
        discre_grid_gpu.free()
        
    def train(self):
        self.initialize()

        # Pre-compute transitions, rewards, and next states for all states and actions
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n
        transition_prob = np.zeros((num_states, num_actions), dtype=np.float32)
        reward_arr = np.zeros((num_states, num_actions, self.reward_dim), dtype=np.float32)
        next_state_arr = np.zeros((num_states, num_actions), dtype=np.int32)

        for state in range(num_states):
            transition_prob[state], reward_arr[state], next_state_arr[state] = self.env.get_transition(state)
        
        # Flatten arrays for GPU transfer
        transition_prob_flat = transition_prob.flatten()
        reward_arr_flat = reward_arr.flatten()
        next_state_arr_flat = next_state_arr.flatten()

        num_Racc_total = self.V.shape[1]  # Get the total number of accumulated rewards
        
        for t in tqdm(range(1, self.time_horizon + 1), desc="Training..."):
            # min, max = 0, np.floor( (self.time_horizon - t) / self.discre_alpha ) * self.discre_alpha
            # time_grid = np.arange(min, max + self.discre_alpha, self.discre_alpha)

            # Use the discretized grid from initialization that accounts for exponential growth
            # Calculate the sum of the geometric series for the remaining timesteps
            remaining_steps = self.time_horizon - t
            if self.gamma == 1:
                max_accumulated_discounted_reward = remaining_steps
            else:
                max_accumulated_discounted_reward = (1 - self.gamma ** (remaining_steps)) / (1 - self.gamma)

            max_possible_reward = min(max_accumulated_discounted_reward, self.discre_grid[-2])
            
            # max_possible_reward = np.round(max_accumulated_discounted_reward / self.discre_alpha) * self.discre_alpha / scaling_factor

            max_possible_discretized_reward = self.discre_func(max_possible_reward)

            while t != self.time_horizon and max_possible_discretized_reward < 1:
                max_possible_discretized_reward = self.discre_func(max_possible_discretized_reward+self.discre_alpha)

            print()
            print(f"Max discretized reward at t = {remaining_steps}: {max_possible_discretized_reward}")
            
            # Filter the discretization grid based on the computed max possible reward
            time_grid = self.discre_grid[self.discre_grid <= max_possible_discretized_reward]
            curr_Racc = [np.asarray(r) for r in list(itertools.product(time_grid, repeat=self.reward_dim))] # Current possible rewards.
            # assert all([len(x) == self.reward_dim for x in curr_Racc]), "Invalid reward accumulation"

            # for Racc in tqdm(curr_Racc, desc="Iterating Racc..."):
            #     Racc_code = self.encode_Racc(Racc)
                
            #     for state in range(self.env.observation_space.n):
            #         # print(f"State: {state}, Racc: {Racc}, Racc_code: {Racc_code}")
            #         transition_prob, reward_arr, next_state_arr = self.env.get_transition(state)  # vectorized, in dimensionality of the action space
                    
            #         next_Racc = Racc + np.power(self.gamma, self.time_horizon - t) * reward_arr  # use broadcasting
            #         # print(f"  Next_Racc: {next_Racc}")
            #         next_Racc_discretized = self.discre_func(next_Racc)  # Discretize next rewards.
            #         # print(f"  Next_Racc_discretized: {next_Racc_discretized}")
            #         next_Racc_code = [self.encode_Racc(d) for d in next_Racc_discretized]
            #         # print(f"  Next_Racc_code: {next_Racc_code}")
                    
            #         all_V = transition_prob * self.V[next_state_arr.astype(int), next_Racc_code, t - 1]
            #         # print(f"  all_V: {all_V}")
                    
            #         # Print the indices being accessed
            #         for a in range(len(transition_prob)):
            #             for nr_code in next_Racc_code:
            #                 idx = next_state_arr[a] * self.V.shape[1] * (self.time_horizon + 1) + nr_code * (self.time_horizon + 1) + (t - 1)
            #                 # print(f"  Accessing V[{next_state_arr[a]}, {nr_code}, {t - 1}] (flattened idx: {idx}) = {self.V[next_state_arr[a], nr_code, t - 1]}")
                    
            #         max_V = np.max(all_V)
            #         best_action = np.argmax(all_V)

            #         # print(f"  max_V: {max_V}, best_action: {best_action}")
                    
            #         update_idx = state * self.V.shape[1] * (self.time_horizon + 1) + Racc_code * (self.time_horizon + 1) + t
            #         # print(f"  Updating V[{state}, {Racc_code}, {t}] (flattened idx: {update_idx}) with {max_V}")
            #         self.V[state, Racc_code, t] = max_V
            #         self.Pi[state, Racc_code, t] = best_action
            
            # self.evaluate()

            num_Racc = len(curr_Racc)
            curr_Racc_np = np.array(curr_Racc, dtype=np.float32).flatten()
            Racc_code = np.array([self.encode_Racc(r) for r in curr_Racc], dtype=np.int32)

            # Print self.V after initialization
            # print("self.V after initialization:")
            # print(self.V)

            # print("discre_grid")
            # print(self.discre_grid.astype(np.float32))

            # print("len(self.discre_grid)")
            # print(len(self.discre_grid))
            
            self.parallel_compute(self.V, self.Pi, transition_prob_flat, reward_arr_flat, next_state_arr_flat, 
                            self.gamma, self.time_horizon, curr_Racc_np, Racc_code.flatten(), 
                            num_states, num_Racc_total, num_Racc, self.reward_dim, self.num_actions, t,
                            self.discre_alpha, self.growth_rate, self.discre_grid.astype(np.float32),
                            len(self.discre_grid))
        
        # print("Finish training")
        # print("self.V")
        # print(self.V)
        # print("self.Pi")
        # print(self.Pi)
        self.evaluate(final=True)
        # np.savez(self.save_path, V=self.V, Pi=self.Pi, Racc_record=self.Racc_record)
    
    def evaluate(self, final=False):
        # self.env.seed(self.seed)
        state = self.env.reset(seed=self.seed)
        # Ensure the renders directory exists within the specified save path
        renders_path = self.save_path + '/renders'
        os.makedirs(renders_path, exist_ok=True)
        img_path = self.save_path + f'/renders/env_render_init.png'
        if isinstance(self.env, envs.Resource_Gathering.ResourceGatheringEnv):
            self.env.render(save_path=img_path)
        Racc = np.zeros(self.reward_dim)
        c = 0
        
        for t in range(self.time_horizon, 0, -1):
            Racc_code = self.encode_Racc(self.discre_func(Racc))
            action = self.Pi[state, Racc_code, t]
            
            next, reward, done = self.env.step(action)
            if isinstance(self.env, envs.Resource_Gathering.ResourceGatheringEnv):
                img_path = self.save_path + f'/renders/env_render_{self.time_horizon-t}.png'
                self.env.render(save_path=img_path)
                # decoded_pos, decoded_status = self.env.decode_state(next)
                # print(f"Decoded Position: {decoded_pos}, Decoded Resource Status: {decoded_status}")
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
        