
#!/bin/bash



export LD_LIBRARY_PATH=/localhome/pnaddaf/anaconda3/envs/env/lib/
for k in "multi" "single"
do
for i in "ACM" "citeseer" 
do
for j in '0' '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1'
do
python -u pn2_main.py --dataSet "$i" --sampling_method "deterministic" --method "$k" --alpha "$j" 
done
done
done