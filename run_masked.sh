
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/

for j in '7'
do
for k in "multi"
do
for i in  "IMDB" "citeseer"
do
for c in '1'
do
python -u pn2_main.py --dataSet "$i" --loss_type "$j" --c "$c" --method "$k"
done
done
done
done