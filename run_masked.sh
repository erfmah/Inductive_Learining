
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/


for i in  "cora" "ACM" "IMDB"
do
for j in '8'
do
python -u pn2_main.py --dataSet "$i" --sampling_method "deterministic" --loss_type "$j"
done
done
