
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/


for i in "ACM" "citeseer" "cora" "imdb"
do
for j in 'Multi_GAT' 'Multi_GCN'
do
python -u pn2_main.py --dataSet "$i" --sampling_method "deterministic"
done
done
