"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for argument parsing.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 13. 02. 2022
"""

import os
from pathlib import Path
from functools import singledispatch

import cv2


def execute_command(command):
    os.system(command)


def create_script(command):
    with open(f'video_script.sh', 'w') as script:
        script.write("# Script automatically created with TCN Dataset Maker.\n")
        script.write("# Author: Daniel Konecny (xkonec75).\n")
        script.write(command)


@singledispatch
def get_seconds_from_frame(arg, frame):
    fps = arg.get(cv2.CAP_PROP_FPS)
    seconds = frame / fps
    return seconds


@get_seconds_from_frame.register
def _(arg: int, frame):
    seconds = frame / arg
    return seconds


@get_seconds_from_frame.register
def _(arg: cv2.VideoCapture, frame):
    fps = arg.get(cv2.CAP_PROP_FPS)
    seconds = frame / fps
    return seconds


@get_seconds_from_frame.register
def _(arg: str, frame):
    cap = cv2.VideoCapture(arg)
    fps = cap.get(cv2.CAP_PROP_FPS)
    seconds = frame / fps
    return seconds


@get_seconds_from_frame.register
def _(arg: Path, frame):
    cap = cv2.VideoCapture(str(arg.resolve()))
    fps = cap.get(cv2.CAP_PROP_FPS)
    seconds = frame / fps
    return seconds


def get_timestamp_from_seconds(seconds):
    milliseconds = int((seconds % 1) * 1000)
    hours = 0
    minutes = 0
    if seconds > 3600:
        hours = int(seconds // 3600)
        seconds %= 3600
    if seconds > 60:
        minutes = int(seconds // 60)
        seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"
