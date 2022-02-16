"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for loading training data and rearranging them for specific training purposes.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 16. 02. 2022
"""

from pathlib import Path
import contextlib

import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils.params import parse_arguments


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

    def get_dataset_size(self, val_split=0.2):
        path = Path(self.directory)
        count = len(list(path.glob('*.png')))

        trn_count = int(round((1 - val_split) * count))
        val_count = int(round(val_split * count))

        return trn_count, val_count

    def get_dataset_generators(self, batch_size=64, val_split=0.2):
        if self.verbose:
            print("DH - Loading train and validation dataset...")

        with contextlib.redirect_stdout(None):
            trn_ds = tf.keras.utils.image_dataset_from_directory(
                self.directory,
                labels=None,
                label_mode=None,
                batch_size=batch_size,
                image_size=(self.steps * self.height, self.cameras * self.width),
                shuffle=False,
                validation_split=val_split,
                subset="training"
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.directory,
                labels=None,
                label_mode=None,
                batch_size=batch_size,
                image_size=(self.steps * self.height, self.cameras * self.width),
                shuffle=False,
                validation_split=val_split,
                subset="validation"
            )

        if self.verbose:
            print(f'DH -- Number of train batches loaded: {tf.data.experimental.cardinality(trn_ds)}.')
            print(f'DH -- Number of validation batches loaded: {tf.data.experimental.cardinality(val_ds)}.')

        trn_ds = trn_ds.shuffle(10000)

        # TODO - optimize loading
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

    def get_random_dataset_generators(self, batch_size=64, val_split=0.2):
        if self.verbose:
            print("DH - Loading train and validation dataset...")

        random_seed = tf.random.uniform(shape=(), minval=1, maxval=2 ** 32, dtype=tf.int64)

        with contextlib.redirect_stdout(None):
            trn_ds = tf.keras.utils.image_dataset_from_directory(
                self.directory,
                labels=None,
                label_mode=None,
                batch_size=batch_size,
                image_size=(self.steps * self.height, self.cameras * self.width),
                shuffle=True,
                seed=random_seed,
                validation_split=val_split,
                subset="training"
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.directory,
                labels=None,
                label_mode=None,
                batch_size=batch_size,
                image_size=(self.steps * self.height, self.cameras * self.width),
                shuffle=True,
                seed=random_seed,
                validation_split=val_split,
                subset="validation"
            )

        if self.verbose:
            print(f'DH -- Number of train batches loaded: {tf.data.experimental.cardinality(trn_ds)}.')
            print(f'DH -- Number of validation batches loaded: {tf.data.experimental.cardinality(val_ds)}.')

        return trn_ds, val_ds


def test():
    args = parse_arguments()

    dataset_handler = DatasetHandler(
        args.location,
        args.steps,
        args.cameras,
        args.height,
        args.width,
        args.verbose
    )

    dataset_size = dataset_handler.get_dataset_size()
    print(dataset_size[0], dataset_size[1])

    trn_ds, val_ds = dataset_handler.get_dataset_generators(args.batch_size, args.val_split)

    for batch in trn_ds:
        for grid in batch:
            plt.imshow(grid / 255.)
            plt.axis("off")
            plt.show()
            break
        break

    for batch in val_ds:
        for grid in batch:
            plt.imshow(grid / 255.)
            plt.axis("off")
            plt.show()
            break
        break


if __name__ == "__main__":
    test()
