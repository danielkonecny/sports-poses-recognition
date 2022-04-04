"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for loading training data and rearranging them for specific training purposes.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 04. 04. 2022
"""

from pathlib import Path

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


def batch_provider(ds, batch_size=30):
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

    def get_dataset_size(self, val_split=0.2):
        path = Path(self.directory)

        img_count = len(list(path.glob('*/cam0/*.png')))
        triplet_count = (img_count - 1) * 6
        train_count = int(round((1 - val_split) * triplet_count))
        val_count = int(round(val_split * triplet_count))

        return train_count, val_count

    def get_scene_dataset(self, scene_dir, val_split=0.2):
        image_count = len(list(scene_dir.glob('cam0/*.png')))
        triplet_count = (image_count - 1) * 6

        if self.verbose:
            print(f"DH --- Image count: {image_count}")
            print(f"DH --- Triplet count: {triplet_count}")

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
            tf.data.Dataset.list_files(str(scene_dir / 'cam0/*.png'), shuffle=False).take(image_count - 1),
            tf.data.Dataset.list_files(str(scene_dir / 'cam1/*.png'), shuffle=False).take(image_count - 1),
            tf.data.Dataset.list_files(str(scene_dir / 'cam2/*.png'), shuffle=False).take(image_count - 1),
            tf.data.Dataset.list_files(str(scene_dir / 'cam0/*.png'), shuffle=False).skip(1),
            tf.data.Dataset.list_files(str(scene_dir / 'cam1/*.png'), shuffle=False).skip(1),
            tf.data.Dataset.list_files(str(scene_dir / 'cam2/*.png'), shuffle=False).skip(1)
        ]

        ds = None
        for config in triplet_configs:
            choice_dataset = tf.data.Dataset.from_tensors(config).repeat(image_count - 1).unbatch()
            new_ds = tf.data.Dataset.choose_from_datasets(list_dss, choice_dataset)
            new_ds = new_ds.batch(3, drop_remainder=True)

            if ds is None:
                ds = new_ds
            else:
                ds = ds.concatenate(new_ds)

        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

        return ds, triplet_count

    def get_dataset(self, val_split):
        if self.verbose:
            print("DH - Loading train and validation dataset...")

        ds = None
        size = 0

        for scene_path in self.directory.glob("*"):
            if self.verbose:
                print(f"DH -- Processing scene from path {scene_path}...")

            new_ds, new_size = self.get_scene_dataset(scene_path, val_split)
            size += new_size
            if ds is None:
                ds = new_ds
            else:
                ds = ds.concatenate(new_ds)

        buffer_size = 256
        ds = ds.shuffle(buffer_size, reshuffle_each_iteration=False)

        val_size = int(round(size * val_split))
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

    train_size, val_size = dataset_handler.get_dataset_size()
    print(train_size, val_size)

    train_ds, val_ds = dataset_handler.get_dataset(args.val_split)

    for epoch in range(1):
        print(f"\nEpoch {epoch}")
        for batch in batch_provider(train_ds, args.batch_size):
            print(batch.shape)

        for batch in batch_provider(val_ds, args.batch_size):
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


if __name__ == "__main__":
    test()
