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


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)

    scene = tf.strings.regex_replace(file_path, r".*/scene(\d+)_cam(\d)_image(\d+).png", r"\1")
    scene = tf.strings.to_number(scene, tf.int32)

    cam = tf.strings.regex_replace(file_path, r".*/scene(\d+)_cam(\d)_image(\d+).png", r"\2")
    cam = tf.strings.to_number(cam, tf.int32)

    index = tf.strings.regex_replace(file_path, r".*/scene(\d+)_cam(\d)_image(\d+).png", r"\3")
    index = tf.strings.to_number(index, tf.int32)

    return img, scene, cam, index


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

    def get_grid_dataset_generators(self, batch_size=64, val_split=0.2):
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

    def get_dataset_generators(self, val_split=0.2):
        if self.verbose:
            print("DH - Loading train and validation dataset...")

        image_count = len(list(self.directory.glob('*.png')))

        list_dss = [
            tf.data.Dataset.list_files(str(self.directory / 'cam0/*.png'), shuffle=False).take(image_count - 1),
            tf.data.Dataset.list_files(str(self.directory / 'cam0/*.png'), shuffle=False).skip(1),
            tf.data.Dataset.list_files(str(self.directory / 'cam1/*.png'), shuffle=False).take(image_count - 1),
            tf.data.Dataset.list_files(str(self.directory / 'cam1/*.png'), shuffle=False).skip(1),
            tf.data.Dataset.list_files(str(self.directory / 'cam2/*.png'), shuffle=False).take(image_count - 1),
            tf.data.Dataset.list_files(str(self.directory / 'cam2/*.png'), shuffle=False).skip(1)
        ]

        dss = []
        for list_ds in list_dss:
            dss.append(list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE))

        tensor = tf.constant([0, 2, 1], dtype=tf.int64)
        choice_dataset = tf.data.Dataset.from_tensors(tensor).repeat(image_count - 1).unbatch()
        ds = tf.data.Dataset.choose_from_datasets(dss, choice_dataset)
        ds = ds.batch(3, drop_remainder=True)

        # ds = ds.shuffle(image_count, reshuffle_each_iteration=False)

        val_size = int(image_count * val_split)
        train_ds = ds.skip(val_size)
        val_ds = ds.take(val_size)

        return train_ds, val_ds


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

    train_ds, val_ds = dataset_handler.get_dataset_generators()

    for epoch in range(1):
        print(f"\nEpoch {epoch}")
        for image, scene, cam, index in train_ds:
            print(f"Train - Image Shape: {image.numpy().shape}, Scene: {scene.numpy()}, "
                  f"Cam: {cam.numpy()}, Index: {index.numpy()}")
            # plt.imshow(image.numpy() / 255.)
            # plt.axis("off")
            # plt.show()

        for image, scene, cam, index in val_ds:
            print(f"Val - Image Shape: {image.numpy().shape}, Scene: {scene.numpy()}, "
                  f"Cam: {cam.numpy()}, Index: {index.numpy()}")
            # plt.imshow(image.numpy() / 255.)
            # plt.axis("off")
            # plt.show()

    # dataset_size = dataset_handler.get_dataset_size()
    # print(dataset_size[0], dataset_size[1])
    #
    # trn_ds, val_ds = dataset_handler.get_grid_dataset_generators(args.batch_size, args.val_split)
    #
    # for batch in trn_ds:
    #     for grid in batch:
    #         plt.imshow(grid / 255.)
    #         plt.axis("off")
    #         plt.show()
    #         break
    #     break
    #
    # for batch in val_ds:
    #     for grid in batch:
    #         plt.imshow(grid / 255.)
    #         plt.axis("off")
    #         plt.show()
    #         break
    #     break


if __name__ == "__main__":
    test()
