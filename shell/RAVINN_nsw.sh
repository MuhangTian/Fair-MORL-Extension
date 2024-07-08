#!/bin/bash
#SBATCH --job-name=RAVI-NN
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-user=nianli_peng@g.harvard.edu
#SBATCH --output=../logs/%j.out
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export env_name=Fair_Taxi_MOMDP
#export size=10
export size=2
#export num_locs=3
export num_locs=1
#export time_horizon=100
export time_horizon=10
#export scaling_factor=13
export scaling_factor=1
export gamma=0.999
export welfare_func_name=nash-welfare
export nsw_lambda=0.0000001
export save_path=$1
export method=ravi_nn
# export seed=$2
export project=RAVI-NN

python ../train.py \
    --env_name $env_name \
    --project $project \
    --size $size \
    --num_locs $num_locs \
    --time_horizon $time_horizon \
    --gamma $gamma \
    --welfare_func_name $welfare_func_name \
    --nsw_lambda $nsw_lambda \
    --save_path $save_path \
    --method $method \
    --scaling_factor $scaling_factor \
    --wandb \