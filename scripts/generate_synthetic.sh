#!/bin/bash

# Iterate over each value in the lists for num_users, num_classes, and num_dimensions
for n_users in 20 60 100 140; do
    for n_classes in 10 20 60 100; do
        for n_dims in 20 60 100 140; do
            # Execute the Python script with the current values
            python scripts/generate_synthetic.py --seed 42 --n_users $n_users --n_classes $n_classes --n_dims $n_dims --iid --output_path data/synth_iid/synthetic_data_nusers_${n_users}_nclasses_${n_classes}_ndims_${n_dims}.json
        done
    done
done
