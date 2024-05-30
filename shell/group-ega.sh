for i in {1..20}
do
    nohup ./ra-ega.sh results/ega-ravi-$i.npz $i > ../logs/ega-ravi-$i
    sleep 60
    # nohup ./wel-ega.sh results/ega-wel-$i.npz $i > ../logs/ega-wel-$i
    # sleep 1
    # nohup ./linear-ega.sh results/ega-linear-$i.npz $i > ../logs/ega-linear-$i
    # sleep 1
    # nohup ./mixture-ega.sh results/ega-mixture-$i.npz $i > ../logs/ega-mixture-$i
    # sleep 1
done
