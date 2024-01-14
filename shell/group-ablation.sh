for alpha in $(seq 0.3 0.05 1.2)
do
    for i in {1..5}
    do
        sbatch shell/ra-ablation.sh results/ravi-ablation-2-$i-$alpha.npz $alpha
        sleep 0.1
    done
done