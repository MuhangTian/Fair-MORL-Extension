for i in {1..20}
do
    # sbatch shell/ra-ega.sh results/ega-ravi-$i.npz
    # sleep 1
    # sbatch shell/wel-ega.sh results/ega-wel-$i.npz
    # sleep 1
    # sbatch shell/linear-ega.sh results/ega-linear-$i.npz
    # sleep 1
    sbatch shell/mixture-ega.sh results/ega-mixture-$i.npz
    sleep 1
done
