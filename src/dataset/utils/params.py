"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for argument parsing.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 17. 06. 2022
"""

from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'location',
        type=str,
        help="Location of the processed data.",
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        default=224,
        help="Dimensions of a training image - height."
    )
    parser.add_argument(
        '-W', '--width',
        type=int,
        default=224,
        help="Dimensions of a training image - width."
    )
    parser.add_argument(
        '-C', '--channels',
        type=int,
        default=3,
        help="Number of channels in used images (e.g. RGB = 3)."
    )
    parser.add_argument(
        '-F', '--framerate',
        type=int,
        default=20,
        help="height of the resulting video.",
    )
    parser.add_argument(
        '-c', '--cameras',
        type=int,
        default=3,
        help="Number of cameras forming the grid of images."
    )
    parser.add_argument(
        '--scene_num',
        type=int,
        default=0,
        help="Number of this dataset scene.",
    )
    parser.add_argument(
        '--cam_num',
        type=int,
        default=0,
        help="Number of this camera in a scene.",
    )
    parser.add_argument(
        '-o', '--overlay',
        type=int,
        default=1000,
        help="Number of frames that have to overlay."
    )
    parser.add_argument(
        '-l', '--load',
        action='store_true',
        help="Use when optical flow has already been calculated and can only be loaded."
    )
    parser.add_argument(
        '--script',
        action='store_true',
        help="When used, videos are not going to be cropped directly but script doing so is going to be created."
    )
    parser.add_argument(
        '-s', '--steps',
        type=int,
        default=3,
        help="Number of steps forming the grid of images."
    )
    parser.add_argument(
        '-t', '--move_thresh',
        type=int,
        default=80,
        help="Threshold for detecting movement in flow between 0 and 100."
    )
    parser.add_argument(
        '-f', '--frame_skip',
        type=int,
        default=7,
        help="Number of frames in between optical flow is calculated."
    )
    parser.add_argument(
        '--export_dir',
        type=str,
        default=".",
        help="Location where the dataset images will be exported in scene*/cam* subdirectories."
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Use to turn on additional text output about what is happening."
    )
    return parser.parse_args()
