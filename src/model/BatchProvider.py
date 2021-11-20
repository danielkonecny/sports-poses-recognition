"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for loading training data and rearranging them for specific training purposes.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 20. 11. 2021
Source: https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes/42298440#42298440 (cubify function)
"""

from pathlib import Path
from argparse import ArgumentParser
import random
import contextlib

import tensorflow as tf
import numpy as np
import cv2
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
    old_shape = np.array(image_grid.shape)
    repeats = (old_shape / new_shape).astype(int)
    tmp_shape = np.column_stack([repeats, new_shape]).ravel()
    order = np.arange(len(tmp_shape))
    order = np.concatenate([order[::2], order[1::2]])
    # new_shape must divide old_shape evenly or else ValueError will be raised
    return image_grid.reshape(tmp_shape).transpose(order).reshape(-1, *new_shape)


def cubify2(image_grid, new_shape=(224, 224, 3)):
    height = new_shape[0]
    width = new_shape[1]
    channels = new_shape[2]
    steps = image_grid.shape[0] // height
    cameras = image_grid.shape[1] // width

    reshaped = np.empty((steps, cameras, height, width, channels))
    for i in range(cameras):
        for j in range(steps):
            reshaped[i][j] = image_grid[i * height:(i + 1) * height, j * width:(j + 1) * width, :]

    return tf.reshape(reshaped, (steps * cameras, height, width, channels))


def triplets_in_grid(grid_shape=(3, 3)):
    steps = grid_shape[0]
    cameras = grid_shape[1]

    # TODO - Use np.meshgrid to create all combinations.

    pass


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

    def load_file_names(self, shuffle=True):
        if self.verbose:
            print("BP - Loading and shuffling of file names...")

        self.file_paths = list((self.directory / 'grids').glob('*.png'))
        self.size = len(self.file_paths)

        if shuffle:
            random.shuffle(self.file_paths)
        else:
            self.file_paths = sorted(self.file_paths)

    def load_files(self):
        """
        Large free memory space is necessary as all loaded grids take up GBs of space.
        """

        if self.verbose:
            print("BP - Loading of grid files...")

        self.files = np.empty((self.size, self.steps * self.height, self.cameras * self.width, self.image_channels))

        for index in range(self.size):
            self.files[index] = cv2.imread(str(self.file_paths[index])) / 255.

        if self.verbose:
            print(f"BP - Grid files loaded with size {self.files.nbytes / 1024 / 1024} MB.")

    def grid_batch_generator(self, grid_batch_size=100):
        for index in range(0, self.size, grid_batch_size):
            if index + grid_batch_size > self.size:
                grid_batch_size = self.size - index

            grid_batch = np.empty(
                (grid_batch_size, self.steps * self.height, self.cameras * self.width, self.image_channels)
            )

            for j in range(grid_batch_size):
                grid_batch[j] = cv2.imread(str(self.file_paths[index + j])) / 255.

            if self.verbose:
                print("BP - New batch of grids loaded.")

            np.random.shuffle(grid_batch)
            yield grid_batch

    def triplet_generator(self, grid):
        # Using fixed steps 0 (t) and 2 (t+14)
        steps = [0, 2]
        step1 = grid[steps[0] * self.height:(steps[0] + 1) * self.height, :, :]
        step2 = grid[steps[1] * self.height:(steps[1] + 1) * self.height, :, :]

        order = np.arange(self.cameras)
        np.random.shuffle(order)

        for shift in range(self.cameras):
            shifted_order = (order + shift) % 3

            anchor = step1[:, shifted_order[0] * self.width:(shifted_order[0] + 1) * self.width, :]
            positive = step1[:, shifted_order[1] * self.width:(shifted_order[1] + 1) * self.width, :]
            negative = step2[:, shifted_order[2] * self.width:(shifted_order[2] + 1) * self.width, :]

            yield [anchor, positive, negative]

    def grids_to_batch(self, grid_batch, group, batch_index, grid_batch_size):
        triplet_size = 3
        permutations = 3

        batch = np.empty(
            (permutations * len(grid_batch), triplet_size, self.height, self.width, self.image_channels)
        )

        index = 0
        for grid in grid_batch:
            for triplet in self.triplet_generator(grid):
                batch[index] = triplet
                index += 1

        np.save(
            self.directory / f'batches/{grid_batch_size:03d}/{group}/scene004_{group}_batch{batch_index:05d}.npy',
            batch
        )
        if self.verbose:
            print(f"BP -- scene004_{group}_batch{batch_index:05d}.npy with shape {batch.shape} saved.")

    def batch_converter(self, grid_batch_size=128, train_percentage=0.8):
        if self.verbose:
            print("BP - Converting batches to NumPy nd-array for faster loading...")

        self.load_file_names()

        Path(self.directory / f'batches/{grid_batch_size:03d}/train').mkdir(parents=True, exist_ok=True)
        Path(self.directory / f'batches/{grid_batch_size:03d}/val').mkdir(parents=True, exist_ok=True)

        batch_index = 0
        for grid_batch in self.grid_batch_generator(grid_batch_size):
            if (batch_index + 1) * grid_batch_size <= train_percentage * self.size:
                self.grids_to_batch(grid_batch, "train", batch_index, grid_batch_size)

            elif batch_index * grid_batch_size <= train_percentage * self.size:
                batch_end = int(train_percentage * self.size - batch_index * grid_batch_size)
                if batch_end != 0:
                    self.grids_to_batch(grid_batch[:batch_end], "train", batch_index, grid_batch_size)

                batch_start = batch_end
                self.grids_to_batch(grid_batch[batch_start:], "val", batch_index, grid_batch_size)

            elif (batch_index + 1) * grid_batch_size > train_percentage * self.size:
                self.grids_to_batch(grid_batch, "val", batch_index, grid_batch_size)

            else:
                print("BP -- Should not happen, seek why if you see this.")

            batch_index += 1

    def batch_generator(self, group, grid_batch_size=128):
        if not Path(self.directory / f'batches/{grid_batch_size:03d}').is_dir():
            self.batch_converter(grid_batch_size)

        for batch_name in Path(self.directory / f'batches/{grid_batch_size:03d}/{group}').glob('*.npy'):
            batch = np.load(batch_name)
            np.random.shuffle(batch)
            yield batch

    def get_dataset_generator(self, batch_size=128, val_split=0.2):
        random_seed = tf.random.uniform(shape=(), minval=1, maxval=2**32, dtype=tf.int64)

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
