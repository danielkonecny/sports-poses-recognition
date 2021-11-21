"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for loading training data and rearranging them for specific training purposes.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 21. 11. 2021
"""

from pathlib import Path
from argparse import ArgumentParser
import contextlib

from itertools import combinations
from math import comb

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Use to turn on additional text output about what is happening."
    )
    return parser.parse_args()


def cubify(image_grid, new_shape=(224, 224, 3)):
    """
    Converts grid of images from one image to 1-D array of all images.
    new_shape must divide old_shape evenly or else ValueError will be raised.
    Source: https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes/42298440#42298440
    """
    old_shape = np.array(image_grid.shape)
    repeats = (old_shape / new_shape).astype(int)
    tmp_shape = np.column_stack([repeats, new_shape]).ravel()
    order = np.arange(len(tmp_shape))
    order = np.concatenate([order[::2], order[1::2]])
    return image_grid.reshape(tmp_shape).transpose(order).reshape(-1, *new_shape)


def triplets_in_grid(grid_shape=(3, 3)):
    steps = grid_shape[0]
    cameras = grid_shape[1]
    triplet_indices = np.empty(((cameras - 1) * steps * comb(cameras, 2), 3), dtype=np.int32)

    index = 0
    for a_p in combinations(range(cameras), 2):
        for n in range(cameras, cameras * steps):
            triplet_indices[index] = [a_p[0], a_p[1], n]
            index += 1

    return triplet_indices


class BatchProvider:
    def __init__(self, directory, cameras, steps, width, height, verbose=False):
        self.directory = Path(directory)
        self.width = width
        self.height = height
        self.cameras = cameras
        self.steps = steps
        self.file_paths = []
        self.files = None
        self.size = 0
        self.image_channels = 3

        self.verbose = verbose

        if self.verbose:
            print("Batch Provider (BP) initialized.")

    def get_dataset_generator(self, batch_size=128, val_split=0.2):
        random_seed = tf.random.uniform(shape=(), minval=1, maxval=2 ** 32, dtype=tf.int64)

        with contextlib.redirect_stdout(None):
            trn_ds = tf.keras.utils.image_dataset_from_directory(
                self.directory,
                labels=None,
                label_mode=None,
                batch_size=batch_size,
                image_size=(self.steps * self.height, self.cameras * self.width),
                validation_split=val_split,
                subset="training",
                seed=random_seed
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.directory,
                labels=None,
                label_mode=None,
                batch_size=batch_size,
                image_size=(self.steps * self.height, self.cameras * self.width),
                validation_split=val_split,
                subset="validation",
                seed=random_seed
            )
        return trn_ds, val_ds


def test():
    args = parse_arguments()

    batch_provider = BatchProvider(
        args.directory,
        args.cameras,
        args.steps,
        args.width,
        args.height,
        verbose=args.verbose
    )

    batch_size = 64

    trn_ds, val_ds = batch_provider.get_dataset_generator(batch_size)

    for batch in trn_ds:
        for nonuplet in batch:
            reshaped = cubify(nonuplet.numpy(), (224, 224, 3))
            print(reshaped.shape)

            for i in range(9):
                plt.imshow(reshaped[i].astype("uint8"))
                plt.axis("off")
                plt.show()
            break
        break


if __name__ == "__main__":
    test()
