#!/bin/bash

n_samples=10000
majority_proportion=0.8
overlap=0.1

# Iterate over each value in the lists for num_users, num_classes, and num_dimensions
for n_users in 20 60 100 140; do
    for n_classes in 10 20 60 100; do
        for n_dims in 20 60 100 140; do
            # Execute the Python script with the current values
            python scripts/generate_synthetic_majority_minority.py --seed 42 --n_samples $n_samples --n_users $n_users --n_classes $n_classes --n_dims $n_dims --majority_proportion $majority_proportion --overlap $overlap --output_path data/synthetic_data_nusers_${n_users}_nclasses_${n_classes}_ndims_${n_dims}_majority_proportion_${majority_proportion}_overlap_${overlap}.json
        done
    done
done
