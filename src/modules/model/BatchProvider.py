"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for loading training data and rearranging them for specific training .
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 07. 11. 2021
"""

import os
from argparse import ArgumentParser
import random

import numpy as np
import cv2


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
    return parser.parse_args()


class BatchProvider:
    def __init__(self, directory, cameras, steps, width, height):
        self.directory = directory
        self.width = width
        self.height = height
        self.cameras = cameras
        self.steps = steps
        self.file_names = []

    def load_file_names(self):
        print("- Loading and shuffling of file names...")
        for file_name in os.listdir(self.directory):
            if file_name.endswith('.png'):
                self.file_names.append(file_name)

        # self.file_names = sorted(self.file_names)
        random.shuffle(self.file_names)

    def get_grid_batch(self, grid_batch_size=100):
        image_channels = 3

        for index in range(0, len(self.file_names), grid_batch_size):
            grid_batch = np.empty(
                (grid_batch_size, self.steps * self.height, self.cameras * self.width, image_channels))
            for j in range(grid_batch_size):
                grid_batch[j] = cv2.imread(f'{self.directory}/{self.file_names[index + j]}') / 255.

            print("- New batch of grids loaded.")
            yield grid_batch

    def get_triplet(self):
        self.load_file_names()

        for grid_batch in self.get_grid_batch(25):
            for grid in grid_batch:
                steps = [0, 2]
                step1 = grid[steps[0] * self.height:(steps[0] + 1) * self.height, :, :]
                step2 = grid[steps[1] * self.height:(steps[1] + 1) * self.height, :, :]
                print(f'-- Using fixed steps {steps}.')

                order = np.arange(self.cameras)
                np.random.shuffle(order)
                print(f'-- Generated new cam order.')

                for shift in range(self.cameras):
                    shifted_order = (order + shift) % 3
                    print(f'-- Shifted cam order: {shifted_order}.')

                    anchor = step1[:, shifted_order[0] * self.width:(shifted_order[0] + 1) * self.width, :]
                    positive = step1[:, shifted_order[1] * self.width:(shifted_order[1] + 1) * self.width, :]
                    negative = step2[:, shifted_order[2] * self.width:(shifted_order[2] + 1) * self.width, :]

                    yield [anchor, positive, negative]


def test():
    args = parse_arguments()

    batch_provider = BatchProvider(args.directory, args.cameras, args.steps, args.width, args.height)

    for anchor, positive, negative in batch_provider.get_triplet():
        cv2.imshow('Anchor Image', anchor)
        cv2.waitKey()
        cv2.imshow('Positive Image', positive)
        cv2.waitKey()
        cv2.imshow('Negative Image', negative)
        cv2.waitKey()


if __name__ == "__main__":
    test()
