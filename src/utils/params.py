"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for argument parsing.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 21. 11. 2021
"""

from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'directory',
        type=str,
        help="Path to the directory with grids of images (without slash at the end).",
    )
    parser.add_argument(
        '-W', '--width',
        type=int,
        default=224,
        help="Dimensions of a training image - width."
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        default=224,
        help="Dimensions of a training image - height."
    )
    parser.add_argument(
        '-C', '--channels',
        type=int,
        default=3,
        help="Number of channels in used images (e.g. RGB = 3)."
    )
    parser.add_argument(
        '-c', '--cameras',
        type=int,
        default=3,
        help="Number of cameras forming the grid of images."
    )
    parser.add_argument(
        '-s', '--steps',
        type=int,
        default=3,
        help="Number of steps forming the grid of images."
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=10,
        help="Number of epochs to be performed on a dataset."
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=64,
        help="Number of triplets in a batch."
    )
    parser.add_argument(
        '-d', '--encoding_dim',
        type=int,
        default=256,
        help="Dimension of latent space in which an image is represented."
    )
    parser.add_argument(
        '-m', '--margin',
        type=float,
        default=0.01,
        help="Margin used for triplet loss - positive has to be at least by a margin closer to anchor than negative."
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Use to turn on additional text output about what is happening."
    )
    return parser.parse_args()
