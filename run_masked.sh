
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/

for i in  "Cora" "photos" "CiteSeer"
do
#for j in '8'
#do
#for a in "Multi_GIN" "Multi_GAT"
#do
#for k in "link"
#do
python -u main.py --dataSet "$i"
#--loss_type "$j" --encoder_type "$a" --query_type "$k"
done
#done
#done
#done

#python  pn2_main.py --dataSet "Cora" --loss_type "5" --encoder_type "Multi_GIN" --query_type "both"