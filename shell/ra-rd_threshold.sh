# # !/bin/bash
# # SBATCH --job-name=RA-VV
# # SBATCH --time=30-00:00:00
# # SBATCH --partition=compsci
# # SBATCH --mem=100G
# # SBATCH --mail-user=muhang.tian@duke.edu
# # SBATCH --output=logs/%j.out
# # SBATCH --mail-type=END
# # SBATCH --mail-type=FAIL

export env_name=ResourceGatheringEnv
export size=5
export num_locs=2
export time_horizon=10
export discre_alpha=0.8
export growth_rate=1.001
export gamma=0.999
export welfare_func_name=resource_damage_scalarization
export threshold=0
export save_path=$1
export method=ra_value_iteration
export scaling_factor=1
export num_resources=3

python ../train.py \
    --env_name $env_name \
    --size $size \
    --num_locs $num_locs \
    --time_horizon $time_horizon \
    --discre_alpha $discre_alpha \
    --growth_rate $growth_rate \
    --gamma $gamma \
    --welfare_func_name $welfare_func_name \
    --threshold $threshold \
    --save_path $save_path \
    --method $method \
    --scaling_factor $scaling_factor \
    --num_resources $num_resources \
    --wandb \