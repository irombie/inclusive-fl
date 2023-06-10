#!/bin/bash

poetry shell
# Parameters
local_epochs=("2" "5")
client_sampling_ratios=("0.4" "1")
mu_fedprox=("0.01" "0.001")
alpha=("0.3" "0.5") 
dataset="cifar"
fl_methods="FedProx"
reweight=("0" "1")

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
					--mu=$mfp
				done
			done
		done
	done
done