export env_name=Fair_Taxi_MOMDP
export size=10
export num_locs=3
export time_horizon=100
export scaling_factor=14
export discre_alpha=1
export growth_rate=1
export gamma=1
export welfare_func_name=egalitarian
export nsw_lambda=0.0000001
export save_path=$1
export method=ra_value_iteration
export p=0.1
export project=Ega
export seed=$2

python ../train.py \
    --env_name $env_name \
    --project $project \
    --seed $2 \
    --scaling_factor $scaling_factor \
    --size $size \
    --num_locs $num_locs \
    --time_horizon $time_horizon \
    --discre_alpha $discre_alpha \
    --gamma $gamma \
    --welfare_func_name $welfare_func_name \
    --nsw_lambda $nsw_lambda \
    --save_path $save_path \
    --method $method \
    --p $p \
    --wandb \
