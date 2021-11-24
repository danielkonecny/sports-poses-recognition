"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for loading training data and rearranging them for specific training purposes.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 24. 11. 2021
"""

from pathlib import Path
import contextlib

from itertools import combinations
from math import comb

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from src.utils.params import parse_arguments


def triplets_in_grid(grid_shape):
    triplet = 3
    steps = grid_shape[0]
    cameras = grid_shape[1]
    triplet_indices = np.empty(((cameras - 1) * steps * comb(cameras, triplet - 1), triplet), dtype=np.int32)

    index = 0
    for a_p in combinations(range(cameras), triplet - 1):
        for n in range(cameras, cameras * steps):
            triplet_indices[index] = [a_p[0], a_p[1], n]
            index += 1

    return triplet_indices


class DatasetHandler:
    def __init__(self, directory, steps, cameras, height, width, verbose=False):
        self.directory = Path(directory)

        self.steps = steps
        self.cameras = cameras
        self.height = height
        self.width = width

        self.verbose = verbose

        if self.verbose:
            print("Dataset Handler (DH) initialized.")

    def get_dataset_generators(self, batch_size=64, val_split=0.2):
        if self.verbose:
            print("DH - Loading train and validation dataset...")

        random_seed = tf.random.uniform(shape=(), minval=1, maxval=2 ** 32, dtype=tf.int64)

        with contextlib.redirect_stdout(None):
            trn_ds = tf.keras.utils.image_dataset_from_directory(
                self.directory / "grids",
                labels=None,
                label_mode=None,
                batch_size=batch_size,
                image_size=(self.steps * self.height, self.cameras * self.width),
                validation_split=val_split,
                subset="training",
                seed=random_seed
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.directory / "grids",
                labels=None,
                label_mode=None,
                batch_size=batch_size,
                image_size=(self.steps * self.height, self.cameras * self.width),
                validation_split=val_split,
                subset="validation",
                seed=random_seed
            )
        print(f'DH -- Number of train batches loaded: {tf.data.experimental.cardinality(trn_ds)}.')
        print(f'DH -- Number of validation batches loaded: {tf.data.experimental.cardinality(val_ds)}.')

        """
        Optimization options:
        - prefetch - no significant improvement noticed
            trn_ds = trn_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        - cache - significant increase of execution time
            trn_ds = trn_ds.cache()
            val_ds = val_ds.cache()
        """

        return trn_ds, val_ds


def test():
    args = parse_arguments()

    dataset_handler = DatasetHandler(
        args.directory,
        args.steps,
        args.cameras,
        args.height,
        args.width,
        args.verbose
    )

    trn_ds, _ = dataset_handler.get_dataset_generators(args.batch_size, args.val_split)

    for batch in trn_ds:
        for grid in batch:
            plt.imshow(grid)
            plt.axis("off")
            plt.show()
            break
        break


if __name__ == "__main__":
    test()
