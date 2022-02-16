"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for training of encoder model - encodes an image to a latent vector representing the sports pose.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 16. 02. 2022
"""

from pathlib import Path
import time
import datetime

import tensorflow as tf
from tensorflow.keras import layers

from itertools import combinations
from math import comb
import numpy as np

from src.model.Cubify import Cubify
from src.model.DatasetHandler import DatasetHandler
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

    return tf.convert_to_tensor(triplet_indices, dtype=tf.int32)


@tf.function
def triplet_loss(triplets, margin=0.01):
    a_p_distance = tf.math.reduce_sum(tf.math.square(tf.math.subtract(triplets[:, 0], triplets[:, 1])), axis=1)
    a_n_distance = tf.math.reduce_sum(tf.math.square(tf.math.subtract(triplets[:, 0], triplets[:, 2])), axis=1)

    loss = tf.math.reduce_sum(tf.math.maximum(0., margin + a_p_distance - a_n_distance))
    accuracy = tf.math.reduce_mean(tf.cast(tf.math.less(a_p_distance, a_n_distance), tf.float32))

    return loss, accuracy


class Encoder:
    def __init__(self, directory, steps, cameras, height, width, channels, encoding_dim, margin,
                 log_dir, ckpt_dir, restore, verbose=False):
        self.directory = Path(directory)

        self.steps = steps
        self.cameras = cameras
        self.height = height
        self.width = width
        self.channels = channels

        self.encoding_dim = encoding_dim
        self.margin = margin

        self.log_dir = Path(log_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.restore = restore

        self.model = self.optimizer = self.ckpt = self.manager = None
        self.trn_loss = tf.keras.metrics.Mean('trn_loss', dtype=tf.float32)
        self.trn_accuracy = tf.keras.metrics.Mean('trn_accuracy', dtype=tf.float32)
        self.val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        self.val_accuracy = tf.keras.metrics.Mean('val_accuracy', dtype=tf.float32)

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.create_file_writer(
            str(self.log_dir / f'{self.current_time}/gradient_tape/train')
        )
        self.val_writer = tf.summary.create_file_writer(
            str(self.log_dir / f'{self.current_time}/gradient_tape/val')
        )

        self.verbose = verbose
        if self.verbose:
            print("Encoder (En) initialized.")

    def create_model(self):
        if self.verbose:
            print("En - Creating encoder model.")

        net_input = tf.keras.Input(shape=(self.height * self.steps, self.width * self.cameras, self.channels))

        "'__call__' method is inherited from 'tf.keras.layers.Layer' and calls defined 'call' method, so no problem."
        # noinspection PyCallingNonCallable
        cubify = Cubify((self.height, self.width, self.channels))(net_input)

        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05, fill_mode='nearest'),
        ])(cubify)

        backbone = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')(data_augmentation)

        head = layers.AveragePooling2D(pool_size=(7, 7))(backbone)
        head = layers.Flatten()(head)
        head = layers.Dense(self.encoding_dim, activation=None, name='trained')(head)
        # Normalize to a vector on a Unit Hypersphere.
        head = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(head)

        self.model = tf.keras.Model(inputs=net_input, outputs=head, name='encoder')

        # Freeze backbone (ResNet50).
        self.model.layers[3].trainable = False

        # Not necessary when only backbone is frozen.
        # self.model.get_layer(name='trained').trainable = True

        self.optimizer = tf.keras.optimizers.Adam()

    def set_checkpoints(self):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=5)

        if self.restore:
            self.ckpt.restore(self.manager.latest_checkpoint)

        if self.verbose and self.restore and self.manager.latest_checkpoint:
            print(f"En -- Checkpoint restored from {self.manager.latest_checkpoint}")
        elif self.verbose:
            print("En -- Checkpoint initialized in ckpts directory.")

    def log_results(self, writer, loss_metrics, acc_metrics, time_metrics, is_train=True):
        with writer.as_default():
            tf.summary.scalar('loss', loss_metrics.result(), step=self.ckpt.step.numpy())
            tf.summary.scalar('accuracy', acc_metrics.result(), step=self.ckpt.step.numpy())
            tf.summary.scalar('time', time_metrics, step=self.ckpt.step.numpy())

        if self.verbose:
            print(f"En --- {'Train' if is_train else 'Val'}:"
                  f" Loss = {loss_metrics.result():.6f},"
                  f" Accuracy = {acc_metrics.result():7.2%}."
                  f" Time/Tuple = {time_metrics:.2f} ms.")

        loss_metrics.reset_states()
        acc_metrics.reset_states()

    @tf.function
    def train_step(self, n_tuple, triplet_indices):
        with tf.GradientTape() as tape:
            n_tuple_expanded = tf.expand_dims(n_tuple, axis=0)
            n_tuple_encoded = self.model(n_tuple_expanded, training=True)
            triplets = tf.gather(n_tuple_encoded, indices=triplet_indices)
            loss, accuracy = triplet_loss(triplets, self.margin)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        self.trn_loss(loss / len(triplet_indices))
        self.trn_accuracy(accuracy)

    def train_on_batches(self, trn_ds, triplet_indices, subdataset_size):
        start_time = time.perf_counter()
        for batch in trn_ds:
            for n_tuple in batch:
                self.train_step(n_tuple, triplet_indices)
        end_time = time.perf_counter()
        tuple_train_time = (end_time - start_time) / subdataset_size * 1e3

        self.log_results(self.train_writer, self.trn_loss, self.trn_accuracy, tuple_train_time)

    @tf.function
    def val_step(self, n_tuple, triplet_indices):
        n_tuple_expanded = tf.expand_dims(n_tuple, axis=0)
        n_tuple_encoded = self.model(n_tuple_expanded, training=False)
        triplets = tf.gather(n_tuple_encoded, indices=triplet_indices)
        loss, accuracy = triplet_loss(triplets, self.margin)

        self.val_loss(loss / len(triplet_indices))
        self.val_accuracy(accuracy)

    def val_on_batches(self, val_ds, triplet_indices, subdataset_size):
        start_time = time.perf_counter()
        for batch in val_ds:
            for n_tuple in batch:
                self.val_step(n_tuple, triplet_indices)
        end_time = time.perf_counter()
        tuple_val_time = (end_time - start_time) / subdataset_size * 1e3

        self.log_results(self.val_writer, self.val_loss, self.val_accuracy, tuple_val_time, False)

    def fit(self, trn_ds, val_ds, epochs, dataset_size):
        if self.verbose:
            print("En - Fitting the model on the training dataset...")

        triplet_indices = triplets_in_grid((self.steps, self.cameras))

        for _ in range(epochs):
            if self.verbose:
                print(f"En -- Epoch {self.ckpt.step.numpy() + 1:02d}.")

            self.train_on_batches(trn_ds, triplet_indices, dataset_size[0])
            self.val_on_batches(val_ds, triplet_indices, dataset_size[1])

            save_path = self.manager.save()

            self.ckpt.step.assign_add(1)

            if self.verbose:
                print(f"En --- Checkpoint saved at {save_path}.")

    def predict(self, images):
        if self.verbose:
            print("En - Predicting images on a model...")

        # TODO - expand dims if only one image

        # TODO - set model to predict mode (cubify and data augmentation turned off)
        encoded_images = self.model(images, training=False)

        # TODO - reduce dims if only one image

        return encoded_images


def main():
    args = parse_arguments()
    dataset_handler = DatasetHandler(
        args.location,
        args.steps,
        args.cameras,
        args.height,
        args.width,
        args.verbose
    )
    encoder = Encoder(
        args.location,
        args.steps,
        args.cameras,
        args.height,
        args.width,
        args.channels,
        args.encoding_dim,
        args.margin,
        args.log_dir,
        args.ckpt_dir,
        args.restore,
        args.verbose
    )
    encoder.create_model()
    encoder.set_checkpoints()

    # encoder.model.summary()
    # encoder.model.layers[3].summary()

    trn_ds, val_ds = dataset_handler.get_dataset_generators(args.batch_size, args.val_split)
    dataset_size = dataset_handler.get_dataset_size()

    encoder.fit(trn_ds, val_ds, args.epochs, dataset_size)


if __name__ == "__main__":
    main()
