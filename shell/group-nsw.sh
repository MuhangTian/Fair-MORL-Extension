for i in {1..20}
do
    # sbatch shell/ra-nsw.sh results/ravi-$i.npz
    # sleep 1
    # sbatch shell/wel-nsw.sh results/wel-$i.npz
    # sleep 1
    # sbatch shell/linear-nsw.sh results/linear-$i.npz
    # sleep 1
    sbatch shell/mixture-nsw.sh results/mixture-$i.npz
    sleep 1
done

