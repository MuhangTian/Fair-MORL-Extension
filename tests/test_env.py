import sys

path = "/Users/mt361/Desktop/Fair-MORL-Extension/Fair-MORL-Extension/"
sys.path.append(path)       # TODO: remove when done
# sys.path.append(input("Enter parent path to the project: "))

import time

import numpy as np
from tqdm import tqdm

from envs.fair_taxi import Fair_Taxi_MOMDP


def get_setting(size: int, num_locs: int):
    """
    To store environment settings

    Parameters
    ----------
    size : int
        size of the grid world in N x N
    num_locs : int
        number of location destination pairs
    """
    if num_locs == 2:
        loc_coords = [[0,0],[3,2]]
        dest_coords = [[0,4],[3,3]]
    elif num_locs == 1:
        loc_coords = [[3,2]]
        dest_coords = [[3,3]]
    elif num_locs == 3:
        loc_coords = [[0,0],[0,5],[3,2]]
        dest_coords = [[0,4],[5,0],[3,3]]
    elif num_locs == 4:
        loc_coords = [[0,0], [0,5], [3,2], [9,0]]
        dest_coords = [[0,4], [5,0], [3,3], [0,9]]
    elif num_locs == 5:
        loc_coords = [[0,0],[0,5],[3,2],[9,0],[4,7]]
        dest_coords = [[0,4],[5,0],[3,3],[0,9],[8,9]]
    else:
        loc_coords = [[0,0],[0,5],[3,2],[9,0],[8,9],[6,7]]
        dest_coords = [[0,4],[5,0],[3,3],[0,9],[4,7],[8,3]]
    return size, loc_coords, dest_coords

def test_normalization(iters=int(1e6), vis=False, printout=False):
    size, loc_coords, dest_coords = get_setting(10,4)
    fuel = 10000
    env = Fair_Taxi_MOMDP(size, loc_coords, dest_coords, fuel, '', 15)
    env.seed(1122)
    env.reset()
    
    for _ in tqdm(range(iters), desc='Testing...'):
        action = np.random.randint(6)
        next, reward, done = env.step(action)   
        if vis:
            env.render() 
        
        if printout:
            print(reward)
        
        if np.sum(reward) not in [0,1]:
            print(reward)
            raise ValueError("Reward not normalized")
        
        if np.sum(reward) == 1:
            pass
        elif np.sum(reward) == 0:
            pass    
        elif np.any(reward) == 1:
            pass
            break
        else:
            raise ValueError(f"Reward not normalized, got reward: {reward}")

def test_interaction(action_list, seconds=0.5):
    size, loc_coords, dest_coords = get_setting(10,4)
    fuel = 10000
    env = Fair_Taxi_MOMDP(size, loc_coords, dest_coords, fuel, '', 15)
    env.seed(1122)
    env.reset()
    
    for action in action_list:
        env.render()
        next, reward, done = env.step(action)
        env.render()
        time.sleep(seconds)
        print(f"Action: {action}, Reward: {reward}, State: {env.decode(next)}")

def test_state_encoding(iters=int(2e7), size=8, objectives=2):
    size, loc_coords, dest_coords = get_setting(size, objectives)
    fuel = 10000
    env = Fair_Taxi_MOMDP(size, loc_coords, dest_coords, fuel, '', 15)
    # env.seed(1122)
    env.reset()
    unique_states, unique_taxi_x, unique_taxi_y, unique_pass_idx = [], [], [], []
    
    for _ in tqdm(range(iters), desc='Testing...'):
        action = np.random.randint(6)
        next, reward, done = env.step(action)
        
        if next not in unique_states:
            unique_states.append(next)
        
        if env.taxi_loc[0] not in unique_taxi_x:
            unique_taxi_x.append(env.taxi_loc[0])
        
        if env.taxi_loc[1] not in unique_taxi_y:
            unique_taxi_y.append(env.taxi_loc[1])
         
        if env.pass_idx not in unique_pass_idx:
            unique_pass_idx.append(env.pass_idx)
    
    print(f"Dest: {env.dest_coords}, length: {len(env.dest_coords)}")
    print(f"Number of unique states: {len(unique_states)}")
    print(f"Number of states: {env.observation_space.n}")
    print(f"Number of unique taxi x: {len(unique_taxi_x)}")
    print(f"Number of unique taxi y: {len(unique_taxi_y)}")
    print(f"Number of unique pass idx: {len(unique_pass_idx)}")
    print("***** CONTENTS *****")
    print(f"Unique states: {np.sort(unique_states)}")
    print(f"Unique taxi x: {np.sort(unique_taxi_x)}")
    print(f"Unique taxi y: {np.sort(unique_taxi_y)}")
    print(f"Unique pass idx: {unique_pass_idx}")
    
    assert all([i == np.sort(unique_states)[i] for i in range(len(unique_states))]), "Check states!"
        

if __name__ == "__main__":
    test_normalization(printout=True)
    # test_interaction([1,1,1,1,1,1,1,1,1,2,2,4,3,3,0,0,0,0,0,0,0,0,0,3,3,3,3,3,3,5])
    # test_state_encoding(size=7, objectives=4)