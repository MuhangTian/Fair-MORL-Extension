import argparse

import wandb
import numpy as np

from algo.ra_value_iteration import RAValueIteration
from algo.welfare_q import WelfareQ
from algo.mixture import MixturePolicy
from algo.linear_scalarize import LinearScalarize
from envs.fair_taxi import Fair_Taxi_MOMDP
from envs.Resource_Gathering import ResourceGatheringEnv
from algo.utils import is_positive_integer, is_positive_float, is_file_not_on_disk, is_within_zero_one_float


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
        loc_coords = [[0,0],[3,2],[1,0]]
        dest_coords = [[0,3],[3,3],[0,1]]
    elif num_locs == 4:
        loc_coords = [[4, 12], [11, 6], [8, 3], [13, 9]]
        dest_coords = [[2, 7], [10, 5], [1, 13], [14, 2]]



    else:
        raise NotImplementedError("Number of locations not implemented")
    
    return size, loc_coords, dest_coords

def parse_arguments():
    prs = argparse.ArgumentParser()
    prs.add_argument('--env_name', choices=["Fair_Taxi_MOMDP", "ResourceGatheringEnv"], required=True)
    prs.add_argument('--size', type=is_positive_integer, default=4)
    prs.add_argument('--num_locs', type=is_positive_integer, default=2)
    prs.add_argument('--time_horizon', type=is_positive_integer, default=1)
    prs.add_argument('--discre_alpha', type=is_positive_float, default=0.5)
    prs.add_argument("--gamma", type=is_positive_float, default=0.999)
    prs.add_argument("--growth_rate", type=is_positive_float, default=1.0)
    prs.add_argument("--welfare_func_name", choices=["egalitarian", "nash-welfare", "p-welfare", "RD-threshold", "Cobb-Douglas"], default="nash-welfare")
    prs.add_argument("--nsw_lambda", type=is_positive_float, default=1e-4)
    prs.add_argument("--wandb", action="store_true")
    prs.add_argument("--save_path", type=is_file_not_on_disk, default="results/trial.npz")
    prs.add_argument("--method", choices=["welfare_q", "ra_value_iteration", "linear_scalarize", "mixture"], required=True)
    prs.add_argument("--lr", type=is_positive_float, default=0.1)
    prs.add_argument("--epsilon", type=is_within_zero_one_float, default=0.1)
    prs.add_argument("--episodes", type=is_positive_integer, default=1000)
    prs.add_argument("--init_val", type=float, default=0.0)
    prs.add_argument("--dim_factor", type=is_positive_float, default=0.9)
    prs.add_argument("--p", type=float, default=0.5)
    prs.add_argument("--threshold", type=is_positive_float, default=4)  # For resource_damage_scalarization
    prs.add_argument("--scaling_factor", type=int, default=1) # scaling factor for accumuated reward initialization in RA-Value Iteration, should be 14 for taxi
    prs.add_argument("--num_resources", type=int, default=8) # number of resources in Resource Gathering
    prs.add_argument("--project", type=str, default="RA-Iteration")
    prs.add_argument("--seed", type=int, default=1122)
    return prs.parse_args()

def init_if_wandb(args):
    if args.wandb:
        # wandb.init(project=args.project, entity="muhang-tian")
        wandb.init(project=args.project, entity="nianli_peng")
        wandb.config.update(args)

if __name__ == "__main__":
    args = parse_arguments()
    if args.env_name == "Fair_Taxi_MOMDP":
        size, loc_coords, dest_coords = get_setting(args.size, args.num_locs)
        env = Fair_Taxi_MOMDP(size, loc_coords, dest_coords, args.time_horizon, '', 15)
    elif args.env_name == "ResourceGatheringEnv":
        env = ResourceGatheringEnv(grid_size=(args.size, args.size),num_resources=args.num_resources,seed=args.seed)
    
    np.random.seed(args.seed)
    env.seed(args.seed)
    
    init_if_wandb(args)
    
    if args.method == "ra_value_iteration":
        algo = RAValueIteration(
            env = env,
            discre_alpha = args.discre_alpha,
            gamma = args.gamma,
            growth_rate = args.growth_rate,
            reward_dim = args.num_locs,
            time_horizon = args.time_horizon,
            welfare_func_name = args.welfare_func_name,
            nsw_lambda = args.nsw_lambda,
            wdb = args.wandb,
            save_path = args.save_path,
            p = args.p,
            threshold = args.threshold,  # Pass threshold for resource_damage_scalarization
            scaling_factor = args.scaling_factor
        )
    elif args.method == "welfare_q":
        algo = WelfareQ(
            env = env,
            lr = args.lr,
            gamma = args.gamma,
            epsilon = args.epsilon,
            episodes = args.episodes,
            init_val = args.init_val,
            welfare_func_name = args.welfare_func_name,
            nsw_lambda = args.nsw_lambda,
            wdb = args.wandb,
            save_path = args.save_path,
            dim_factor = args.dim_factor,
            p = args.p,
        )
    elif args.method == "linear_scalarize":
        algo = LinearScalarize(
            env = env,
            init_val = args.init_val,
            episodes = args.episodes,
            weights = [0.37, 0.63],     # optimal tuned weights
            lr = args.lr,
            gamma = args.gamma,
            epsilon = args.epsilon,
            welfare_func_name = args.welfare_func_name,
            save_path = args.save_path,
            nsw_lambda = args.nsw_lambda,
            p = args.p,
            wdb = args.wandb,
        )
    elif args.method == "mixture":
        algo = MixturePolicy(
            env = env,
            episodes = args.episodes,
            time_horizon = args.time_horizon,
            lr = args.lr,
            epsilon = args.epsilon,
            gamma = args.gamma,
            init_val = args.init_val,
            weights = [[0.21, 0.79],[1.0, 0.0]],     # optimal tuned weights
            interval = 1,   # change policy after t/d timesteps
            welfare_func_name = args.welfare_func_name,
            save_path = args.save_path,
            nsw_lambda = args.nsw_lambda,
            p = args.p,
            wdb = args.wandb,
        )

    algo.train()