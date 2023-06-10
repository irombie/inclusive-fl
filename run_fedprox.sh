#!/bin/bash

# Parameters
local_epochs=("5")
client_sampling_ratios=("0.4" "1")
mu_fedprox=("0.001")
alpha=("0.3" "0.5") # Set to 1 for iid 
dataset="cifar"
fl_method="FedProx"
reweight=("0")

max_parallel_tasks=5  # Maximum number of background tasks
current_tasks=0  # Current number of background tasks

for le in ${local_epochs[@]}; do
	for csr in ${client_sampling_ratios[@]}; do
		for mfp in ${mu_fedprox[@]}; do
			for a in ${alpha[@]}; do
				for rw in ${reweight[@]}; do
					iid=1
					if [ $a != "1" ]; then
						iid=0
					fi
					
					echo "Running with Local Epochs: $le, Client Sampling Ratio: $csr, Mu Fedprox: $mfp, Alpha: $a, Dataset: $dataset, FL Method: $fm, Reweight: $rw"
					
					python src/federated_main.py \
					--model=cnn \
					--dataset=$dataset \
					--gpu=0 \
					--iid=$iid \
					--epochs=10 \
					--lr=0.01 \
					--verbose=1\
					--seed=1 \
					--num_users=100 \
					--frac=$csr \
					--local_ep=$le \
					--local_bs=10 \
					--unequal=0 \
					--dist_noniid=$a \
					--fl_method=$fl_method \
					--mu=$mfp &

					if [[ $current_tasks -eq $max_parallel_tasks ]]; then
						wait  # Wait for the background tasks to finish
						current_tasks=0  # Reset the task counter
					fi
				done
			done
		done
	done
done
wait