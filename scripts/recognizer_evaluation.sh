#!/usr/bin/env bash

# Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
# Script for launching a model training with different parameters to evaluate it.
# Organisation: Brno University of Technology - Faculty of Information Technology
# Author: Daniel Konecny (xkonec75)
# Date: 05. 07. 2022

for split in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.825 0.85 0.875 0.9 0.925 0.95 0.975
do
    for seed in {0..9}
    do
        python3 src/model/Recognizer.py fit "data/upper_body/labeled_directions" "ckpts/encoder/best" -e 20 -b 128 -s "$split" -S "$seed" -E -v -g
    done
done
