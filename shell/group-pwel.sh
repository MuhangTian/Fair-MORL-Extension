for i in {1..20}
do
    # sbatch shell/ra-pwel.sh results/pwel-ravi-$i.npz
    # sleep 1
    sbatch shell/wel-pwel.sh results/pwel-wel-$i.npz
    sleep 1
done
