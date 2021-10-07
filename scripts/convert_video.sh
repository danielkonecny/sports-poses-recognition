#!/usr/bin/env bash

# Input Video Information
in_w=1920
in_h=1080
in_dir=../../dip-data/in/

# Output Video Information
out_dir=../../dip-data/out/

# Crop Settings (computes values to crop the center of the video)
crop_w=960
crop_h=960
crop_start_w=$in_w/2-$crop_w/2
crop_start_h=$in_h/2-$crop_h/2

# Scale Settings - -1 for keeping the same ratio
scale_w=320
scale_h=-1

# Framerate Settings
fps=30

# Audio handling
# -an - removes audio
# -codec:a copy - copies audio

for file in "$in_dir"*.*;
do
    out_file=${file##*/}
    out_file=${out_file%.*}
    out_suffix=${file##*.}
    ffmpeg \
        -i "$file" \
        -filter:v "crop=$crop_w:$crop_h:$crop_start_w:$crop_start_h, fps=$fps, scale=$scale_w:$scale_h" \
        -an \
        "$out_dir""$out_file"_ffmpeged."$out_suffix"
done

