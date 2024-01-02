#!/bin/bash
#SBATCH --job-name=RA-VV
#SBATCH --time=30-00:00:00
#SBATCH --partition=compsci
#SBATCH --mem=100G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=logs/%j.out
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export size=4
export num_locs=2
export time_horizon=100
export discre_alpha=0.8
export gamma=0.999
export welfare_func_name=nsw
export nsw_lambda=0.0000001
export save_path=$1
export method=ra_value_iteration

python train.py \
    --size $size \
    --num_locs $num_locs \
    --time_horizon $time_horizon \
    --discre_alpha $discre_alpha \
    --gamma $gamma \
    --welfare_func_name $welfare_func_name \
    --nsw_lambda $nsw_lambda \
    --save_path $save_path \
    --method $method \
    --wandb \
