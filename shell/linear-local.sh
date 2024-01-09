export size=4
export num_locs=2
export lr=0.1
export gamma=0.999
export epsilon=0.9999
export episodes=100000
export init_val=1
export welfare_func_name=nsw
export nsw_lambda=0.0000001
export save_path=results/welfareq-01.npz
export method=welfare_q
export dim_factor=0.0001
export time_horizon=100

# python -m debugpy --listen linux51:5678 --wait-for-client \
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
    # --wandb \
