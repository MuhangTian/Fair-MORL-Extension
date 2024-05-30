# # !/bin/bash
# # SBATCH --job-name=WelfareQ
# # SBATCH --time=30-00:00:00
# # SBATCH --partition=compsci
# # SBATCH --mem=100G
# # SBATCH --mail-user=muhang.tian@duke.edu
# # SBATCH --output=logs/%j.out
# # SBATCH --mail-type=END
# # SBATCH --mail-type=FAIL

export env_name=Fair_Taxi_MOMDP
export size=10
export num_locs=3
export time_horizon=100
export discre_alpha=1
export growth_rate=1
export gamma=1
export lr=0.1
export epsilon=0.9999
export episodes=2000000
export init_val=1
export welfare_func_name=nash-welfare
export nsw_lambda=0.0000001
export save_path=$1
export method=welfare_q
export dim_factor=0.0001
export seed=$2
export project=NSW

python \
    ../train.py \
    --env_name $env_name \
    --project $project \
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
    --seed $2 \
    --wandb \