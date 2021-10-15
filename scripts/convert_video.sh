#!/usr/bin/env bash

# Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
# Script for easy converting of multiple videos to format suitable for training dataset.
# Organisation: Brno University of Technology - Faculty of Information Technology
# Author: Daniel Konecny (xkonec75)
# Date: 14. 10. 2021

# Input Video Information
in_w=1920
in_h=1080
in_dir=../../dip-data/in/

# Output Video Information
out_dir=../../dip-data/out/

# Crop Settings (computes values to crop the center of the video)
crop_w=1080
crop_h=1080
crop_start_w=$in_w/2-$crop_w/2
crop_start_h=$in_h/2-$crop_h/2

# Scale Settings - -1 for keeping the same ratio
scale_w=224
scale_h=-1

# Framerate Settings
fps=20

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
        -filter:v "crop=$crop_w:$crop_h:$crop_start_w:$crop_start_h, scale=$scale_w:$scale_h, fps=$fps" \
        -an \
        "$out_dir""$out_file"_normalized."$out_suffix"
done

# OTHER USEFUL FFMPEG COMMANDS

# Lossless conversion of MTS to MP4
# ffmpeg -i input.MTS -c:v copy -c:a aac -strict experimental -b:a 128k output.mp4

# Concatenation of file not supporting file-level concatenation (MP4)
# touch videos.txt
# insert word "file" + path to video file
    #file '/path/to/file1'
    #file '/path/to/file2'
    #file '/path/to/file3'
# ffmpeg -f concat -safe 0 -i videos.txt -c copy output.mp4

# Quality of conversion (crf) - 0=lossless - 23=default - 51=worst quality
# ffmpeg -c:v libx264 -crf 0

# Cut videos by timestamp (with re-encoding to have a correct starting frame).
# ffmpeg -ss 00:00:00.000 -i input.mp4 -t 00:00:00.000 -c:v libx264 -crf 0 output.mp4

# Place videos next to each other
# - Two videos:
# ffmpeg -i left.mp4 -i right.mp4 -filter_complex hstack -c:v libx264 -crf 0 output.mp4
# - Three videos:
# ffmpeg -i input0.mp4 -i input1.mp4 -i input2.mp4 -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]" -c:v libx264 -crf 0 output.mp4
# - Four videos:
# ffmpeg -i input0.mp4 -i input1.mp4 -i input2.mp4 -i input3.mp4 -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" -c:v libx264 -crf 0 output.mp4
# - Other options:
# https://stackoverflow.com/questions/11552565/vertically-or-horizontally-stack-mosaic-several-videos-using-ffmpeg
