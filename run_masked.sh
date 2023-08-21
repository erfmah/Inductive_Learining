
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/

for k in "single" "subgraph"
do
for i in  "cora" "computers"
do
for j in '7' '8' '9'
do
for a in "Multi_GAT" "Multi_GAT_2" "Multi_GCN" "Multi_GCN_2"
do
python -u pn2_main.py --dataSet "$i" --loss_type "$j" --method "$k" --encoder_type "$a"
done
done
done
done
