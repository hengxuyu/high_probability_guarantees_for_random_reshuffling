#!/bin/bash
# Activate the Python environment 'hxt'
source /home/luoqijun/anaconda3/bin/activate hxt
# Set the CUDA visible devices
export CUDA_VISIBLE_DEVICES=0
export REPETITION=1

# Loop over seeds and run the Python script
for paralell in {0..19}; do
    export PARALELL=$paralell
    # Run the script with nohup and timestamped log file
    nohup python ../rr_vs_sgd.py > "output_process${paralell}.log" 2> "error_process${paralell}.log" &
done

python ./plot_rr_vs_sgd.py
