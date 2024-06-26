# # !/bin/bash
# # SBATCH --job-name=RA-VV
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
export scaling_factor=13
export discre_alpha=0.8
export growth_rate=1.001
export gamma=0.999
export welfare_func_name=nash-welfare
export nsw_lambda=0.0000001
export save_path=$1
export method=ra_value_iteration
export seed=$2
export project=NSW

python ../train.py \
    --env_name $env_name \
    --project $project \
    --seed $2 \
    --size $size \
    --num_locs $num_locs \
    --time_horizon $time_horizon \
    --discre_alpha $discre_alpha \
    --growth_rate $growth_rate \
    --gamma $gamma \
    --welfare_func_name $welfare_func_name \
    --nsw_lambda $nsw_lambda \
    --save_path $save_path \
    --method $method \
    --scaling_factor $scaling_factor \
    --wandb \