import argparse

import wandb

from algo.ra_value_iteration import RAValueIteration
from algo.welfare_q import WelfareQ
from envs.fair_taxi import Fair_Taxi_MOMDP
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
        loc_coords = [[0,0],[3,2],[1,0],[2,2]]
        dest_coords = [[0,3],[3,3],[0,1],[2,0]]
    else:
        raise NotImplementedError("Number of locations not implemented")
    
    return size, loc_coords, dest_coords

def parse_arguments():
    prs = argparse.ArgumentParser()
    prs.add_argument('--size', type=is_positive_integer, default=4)
    prs.add_argument('--num_locs', type=is_positive_integer, default=2)
    prs.add_argument('--time_horizon', type=is_positive_integer, default=1)
    prs.add_argument('--discre_alpha', type=is_within_zero_one_float, default=0.5)
    prs.add_argument("--gamma", type=is_within_zero_one_float, default=0.999)
    prs.add_argument("--welfare_func_name", choices=["egalitarian", "nsw", "p-welfare"], default="nsw")
    prs.add_argument("--nsw_lambda", type=is_positive_float, default=1e-4)
    prs.add_argument("--wandb", action="store_true")
    prs.add_argument("--save_path", type=is_file_not_on_disk, default="results/trial.npz")
    prs.add_argument("--method", choices=["welfare_q", "ra_value_iteration"], required=True)
    prs.add_argument("--lr", type=is_positive_float, default=0.1)
    prs.add_argument("--epsilon", type=is_within_zero_one_float, default=0.1)
    prs.add_argument("--episodes", type=is_positive_integer, default=1000)
    prs.add_argument("--init_val", type=float, default=0.0)
    prs.add_argument("--dim_factor", type=is_positive_float, default=0.9)
    prs.add_argument("--p", type=float, default=0.5)
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
    
    if args.method == "ra_value_iteration":
        algo = RAValueIteration(
            env = env,
            discre_alpha = args.discre_alpha,
            gamma = args.gamma,
            reward_dim = args.num_locs,
            time_horizon = args.time_horizon,
            welfare_func_name = args.welfare_func_name,
            nsw_lambda = args.nsw_lambda,
            wdb = args.wandb,
            save_path = args.save_path,
            p = args.p,
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

    algo.train()