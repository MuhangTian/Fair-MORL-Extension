export env_name="Fair_Taxi_MOMDP"
#export size=10
export size=2
#export num_locs=3
export num_locs=1
#export time_horizon=100
export time_horizon=10
#export scaling_factor=13
export scaling_factor=1
export gamma=0.999
export welfare_func_name="nash-welfare"
export nsw_lambda=0.0000001
export save_path=trial.pt
export method="ravi_nn"
# export seed=$2
export project="RAVI-NN"

# python -m debugpy --listen linux55:5678 --wait-for-client \
python \
    train.py \
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
    # --wandb \