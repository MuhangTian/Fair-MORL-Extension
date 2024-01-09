#!/bin/bash
#SBATCH --job-name=LinearS
#SBATCH --time=30-00:00:00
#SBATCH --partition=compsci
#SBATCH --mem=100G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=logs/%j.out
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export size=4
export num_locs=2
export lr=0.1
export gamma=0.999
export epsilon=0.1
export episodes=20000
export init_val=1
export welfare_func_name=egalitarian
export nsw_lambda=0.0000001
export save_path=$1
export method=linear_scalarize
export dim_factor=0.0001
export time_horizon=100
export p=0.1

python \
    train.py \
    --size $size \
    --num_locs $num_locs \
    --lr $lr \
    --gamma $gamma \
    --epsilon $epsilon \
    --episodes $episodes \
    --init_val $init_val \
    --welfare_func_name $welfare_func_name \
    --nsw_lambda $nsw_lambda \
    --save_path $save_path \
    --method $method \
    --dim_factor $dim_factor \
    --time_horizon $time_horizon \
    --p $p \
    --wandb \