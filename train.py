import argparse

import wandb

from algo.ra_value_iteration import RAValueIteration
from envs.fair_taxi import Fair_Taxi_MOMDP


def get_setting(size, num_locs):
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
        dest_coords = [[0,3],[3,3]]
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

def parse_arguments():
    prs = argparse.ArgumentParser()
    prs.add_argument('--size', type=int, default=4)
    prs.add_argument('--num_locs', type=int, default=2)
    prs.add_argument('--time_horizon', type=int, default=100)
    prs.add_argument('--discre_alpha', type=float, default=0.5)
    prs.add_argument("--gamma", type=float, default=0.999)
    prs.add_argument("--welfare_func_name", type=str, default="nsw")
    prs.add_argument("--nsw_lambda", type=float, default=1e-4)
    prs.add_argument("--wandb", action="store_false")
    return prs.parse_args()

def init_if_wandb(args):
    if args.wandb:
        wandb.init(project="RA-Iteration", entity="muhang-tian")
        wandb.config.update(args)

if __name__ == "__main__":
    args = parse_arguments()
    size, loc_coords, dest_coords = get_setting(args.size, args.num_locs)
    env = Fair_Taxi_MOMDP(size, loc_coords, dest_coords, args.time_horizon, '', 15)
    
    init_if_wandb(args)
    
    algo = RAValueIteration(
        env = env,
        discre_alpha = args.discre_alpha,
        gamma = args.gamma,
        reward_dim = args.num_locs,
        time_horizon = args.time_horizon,
        welfare_func_name = args.welfare_func_name,
        nsw_lambda = args.nsw_lambda,
        wdb = args.wandb
    )
    algo.train()
    algo.evaluate()