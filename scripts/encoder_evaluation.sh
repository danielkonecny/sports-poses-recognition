#!/usr/bin/env bash

# Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
# Script for launching a model training with different parameters to evaluate it.
# Organisation: Brno University of Technology - Faculty of Information Technology
# Author: Daniel Konecny (xkonec75)
# Date: 26. 06. 2022

for dimension in 64 128 256 512
do
    for seed in {0..4}
    do
        mkdir logs/latent-$dimension-"$seed"

        python3 src/model/Encoder.py "data/upper_body" -e 50 -f 0 -m 0.1 -b 64 -s 0.1 -d "$dimension" -S "$seed" -E \
        -ed ckpts/encoder/latent-$dimension-"$seed" -ld logs/latent-$dimension-"$seed" -v -g \
        2> logs/latent-$dimension-"$seed"/err.log | tee logs/latent-$dimension-"$seed"/out.log
    done
done
