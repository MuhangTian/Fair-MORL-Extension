for i in {1..20}
do
    nohup ./ra-nsw.sh ../results/ravi-$i.npz $i > ../logs/ravi-$i &
    sleep 60
    # nohup ./wel-nsw.sh ../results/wel-$i.npz $i > ../logs/wel-$i &
    # sleep 1
    # nohup ./linear-nsw.sh ../results/linear-$i.npz $i > ../logs/linear-$i &
    # sleep 1
    # nohup ./mixture-nsw.sh ../results/mixture-$i.npz $i > ../logs/mixture-$i &
    # sleep 1
done

