#!/bin/bash

poetry shell
# Parameters
local_epochs=("2" "5")
client_sampling_ratios=("0.4" "1")
alpha=("0.3") 
dataset="fmnist"
fl_method="FedAvg"
reweight=("0")

for le in ${local_epochs[@]}; do
	for csr in ${client_sampling_ratios[@]}; do
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
				--fl_method=$fl_method
			done
		done
	done
done