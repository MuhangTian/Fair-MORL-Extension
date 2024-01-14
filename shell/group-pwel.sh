for i in {1..20}
do
    sbatch shell/ra-pwel.sh results/pwel-09-ravi-$i.npz
    sleep 1
    sbatch shell/wel-pwel.sh results/pwel-09-wel-$i.npz
    sleep 1
    sbatch shell/linear-pwel.sh results/pwel-09-linear-$i.npz
    sleep 1
    sbatch shell/mixture-pwel.sh results/pwel-09-mixture-$i.npz
    sleep 1
done
