
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/


for i in  "cora" "ACM" "IMDB" "citeseer"
do
for j in '1' '7'
do
for b in '1' '10' '50' '100'
do
for c in '1' '2' '3' '4' '5'
do
python -u pn2_main.py --dataSet "$i" --sampling_method "deterministic" --loss_type "$j" --b "$b" --c "$c"
done
done
done
done
