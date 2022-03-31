"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for loading training data and rearranging them for specific training purposes.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 31. 03. 2022
"""

from pathlib import Path
import contextlib

import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils.params import parse_arguments


def load_metadata(file_path):
    scene = tf.strings.regex_replace(file_path, r".*/scene(\d+)_cam(\d)_image(\d+).png", r"\1")
    scene = tf.strings.to_number(scene, tf.int32)

    cam = tf.strings.regex_replace(file_path, r".*/scene(\d+)_cam(\d)_image(\d+).png", r"\2")
    cam = tf.strings.to_number(cam, tf.int32)

    index = tf.strings.regex_replace(file_path, r".*/scene(\d+)_cam(\d)_image(\d+).png", r"\3")
    index = tf.strings.to_number(index, tf.int32)

    return file_path, scene, cam, index


def load_image(paths):
    img_anchor = tf.io.read_file(paths[0])
    img_anchor = tf.image.decode_png(img_anchor, channels=3)

    img_positive = tf.io.read_file(paths[1])
    img_positive = tf.image.decode_png(img_positive, channels=3)

    img_negative = tf.io.read_file(paths[2])
    img_negative = tf.image.decode_png(img_negative, channels=3)

    return img_anchor, img_positive, img_negative


def split_ds_into_batches_gen(ds, batch_size=30):
    batch = []
    for i, (a, p, n) in enumerate(ds):
        batch.append([a, p, n])
        if (i + 1) % batch_size == 0:
            yield tf.stack(batch)
            batch = []

    yield tf.stack(batch)


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

    def get_grid_dataset_size(self, val_split=0.2):
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

    def get_triplet_generators(self, val_split=0.2):
        if self.verbose:
            print("DH - Loading train and validation dataset...")

        image_count = len(list(self.directory.glob('*/*.png')))

        cam_count = 3
        triplet_configs = [
            tf.constant([0, 1, 0 + cam_count], dtype=tf.int64),
            tf.constant([0, 2, 0 + cam_count], dtype=tf.int64),
            tf.constant([1, 0, 1 + cam_count], dtype=tf.int64),
            tf.constant([1, 2, 1 + cam_count], dtype=tf.int64),
            tf.constant([2, 0, 2 + cam_count], dtype=tf.int64),
            tf.constant([2, 1, 2 + cam_count], dtype=tf.int64)
        ]

        list_dss = [
            tf.data.Dataset.list_files(str(self.directory / 'cam0/*.png'), shuffle=False).take(image_count - 1),
            tf.data.Dataset.list_files(str(self.directory / 'cam1/*.png'), shuffle=False).take(image_count - 1),
            tf.data.Dataset.list_files(str(self.directory / 'cam2/*.png'), shuffle=False).take(image_count - 1),
            tf.data.Dataset.list_files(str(self.directory / 'cam0/*.png'), shuffle=False).skip(1),
            tf.data.Dataset.list_files(str(self.directory / 'cam1/*.png'), shuffle=False).skip(1),
            tf.data.Dataset.list_files(str(self.directory / 'cam2/*.png'), shuffle=False).skip(1)
        ]

        ds = None
        for config in triplet_configs:
            choice_dataset = tf.data.Dataset.from_tensors(config).repeat(image_count - 1).unbatch()
            tmp_ds = tf.data.Dataset.choose_from_datasets(list_dss, choice_dataset)
            tmp_ds = tmp_ds.batch(3, drop_remainder=True)

            if ds is not None:
                ds = ds.concatenate(tmp_ds)
            else:
                ds = tmp_ds

        ds = ds.shuffle(image_count, reshuffle_each_iteration=False)

        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

        val_size = int(image_count * val_split)
        train_ds = ds.skip(val_size)
        val_ds = ds.take(val_size)

        return train_ds, val_ds

    def get_dataset_generators(self, batch_size=64, val_split=0.2):
        train_ds, val_ds = self.get_triplet_generators(val_split)

        # TODO - wrap the generator in a tf.data.Dataset pipeline
        # train_ds = tf.data.Dataset.from_generator(
        #     split_ds_into_batches_gen,
        #     args=(train_ds, batch_size),
        #     output_signature=(
        #         tf.RaggedTensorSpec(shape=(None, 3, 224, 224, 3), dtype=tf.int32)
        #     )
        # )
        # val_ds = tf.data.Dataset.from_generator(
        #     split_ds_into_batches_gen,
        #     args=(val_ds, batch_size),
        #     output_signature=(
        #         tf.RaggedTensorSpec(shape=(None, 3, 224, 224, 3), dtype=tf.int32)
        #     )
        # )

        train_ds = split_ds_into_batches_gen(train_ds, batch_size)
        val_ds = split_ds_into_batches_gen(val_ds, batch_size)

        return train_ds, val_ds


def test_old():
    args = parse_arguments()

    dataset_handler = DatasetHandler(
        args.location,
        args.steps,
        args.cameras,
        args.height,
        args.width,
        args.verbose
    )

    dataset_size = dataset_handler.get_grid_dataset_size()
    print(dataset_size[0], dataset_size[1])

    trn_ds, val_ds = dataset_handler.get_grid_dataset_generators(args.batch_size, args.val_split)

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

    train_ds, val_ds = dataset_handler.get_dataset_generators(args.batch_size, args.val_split)

    for epoch in range(1):
        print(f"\nEpoch {epoch}")
        for batch in train_ds:
            print(batch.shape)
            for a, p, n in batch:
                plt.imshow(a.numpy() / 255.)
                plt.axis("off")
                plt.show()
                plt.imshow(p.numpy() / 255.)
                plt.axis("off")
                plt.show()
                plt.imshow(n.numpy() / 255.)
                plt.axis("off")
                plt.show()
                break
            break

        for batch in val_ds:
            for a, p, n in batch:
                plt.imshow(a.numpy() / 255.)
                plt.axis("off")
                plt.show()
                plt.imshow(p.numpy() / 255.)
                plt.axis("off")
                plt.show()
                plt.imshow(n.numpy() / 255.)
                plt.axis("off")
                plt.show()
                break
            break


if __name__ == "__main__":
    test()
