export env_name=Fair_Taxi_MOMDP
export size=15
export num_locs=4
export time_horizon=150
export discre_alpha=1
export growth_rate=1
export gamma=1
export lr=0.1
export epsilon=0.1
export episodes=20000
export init_val=1
export welfare_func_name=egalitarian
export nsw_lambda=0.0000001
export save_path=$1
export method=linear_scalarize
export dim_factor=0.0001
export p=0.1
export seed=$2
export project=Ega

python \
    ../train.py \
    --env_name $env_name \
    --project $project \
    --seed $2 \
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