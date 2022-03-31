"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for training of encoder model - encodes an image to a latent vector representing the sports pose.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 08. 03. 2022
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

        # backbone = tf.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet')(data_augmentation)
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
        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=100)

        if self.restore:
            self.ckpt.restore(self.manager.latest_checkpoint)
            self.ckpt.step.assign_add(1)

        if self.verbose and self.restore and self.manager.latest_checkpoint:
            print(f"En -- Checkpoint restored from {self.manager.latest_checkpoint}")
        elif self.verbose:
            print("En -- Checkpoint initialized in ckpts directory.")

    def log_results(self, writer, loss_metrics, acc_metrics, time_metrics, is_train=True):
        loss = loss_metrics.result()
        acc = acc_metrics.result()

        with writer.as_default():
            tf.summary.scalar('loss', loss, step=self.ckpt.step.numpy())
            tf.summary.scalar('accuracy', acc, step=self.ckpt.step.numpy())
            tf.summary.scalar('time', time_metrics, step=self.ckpt.step.numpy())

        if self.verbose:
            print(f"En --- {'Train' if is_train else 'Val'}:"
                  f" Loss = {loss:.6f},"
                  f" Accuracy = {acc:7.2%}."
                  f" Time/Tuple = {time_metrics:.2f} ms.")

        loss_metrics.reset_states()
        acc_metrics.reset_states()

        return loss, acc

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

        loss, acc = self.log_results(self.train_writer, self.trn_loss, self.trn_accuracy, tuple_train_time)

        return loss, acc

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

        loss, acc = self.log_results(self.val_writer, self.val_loss, self.val_accuracy, tuple_val_time, False)

        return loss, acc

    def fit(self, trn_ds, val_ds, epochs, dataset_size):
        if self.verbose:
            print("En - Fitting the model...")

        best_acc = 0
        best_epoch = -1
        best_path = ""

        triplet_indices = triplets_in_grid((self.steps, self.cameras))

        for index in range(epochs):
            if self.verbose:
                print(f"En -- Epoch {self.ckpt.step.numpy() + 1:02d}.")

            self.train_on_batches(trn_ds, triplet_indices, dataset_size[0])
            _, accuracy = self.val_on_batches(val_ds, triplet_indices, dataset_size[1])

            save_path = self.manager.save()

            self.ckpt.step.assign_add(1)

            if accuracy > best_acc:
                best_epoch = index
                best_acc = accuracy
                best_path = save_path

            if self.verbose:
                print(f"En --- Checkpoint saved at {save_path}.")

        return best_epoch, best_path

    def fine_tune(self, trn_ds, val_ds, epochs, dataset_size, best_path=None):
        if self.verbose:
            print("En - Fine tuning the model...")

        if best_path is not None:
            self.ckpt.restore(best_path)
            self.ckpt.step.assign_add(1)
            if self.verbose:
                print(f"En -- Restored best model from path '{best_path}'.")

        self.model.layers[3].trainable = True
        self.optimizer = tf.keras.optimizers.Adam(1e-5)

        best_epoch, best_path = self.fit(trn_ds, val_ds, epochs, dataset_size)

        return best_epoch, best_path

    def encode(self, images):
        if self.verbose:
            print("En - Encoding images with the model...")

        # TODO - set model to predict mode (cubify and data augmentation turned off)
        encoded_images = self.model(images, training=False)

        return encoded_images


def encode():
    args = parse_arguments()
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

    # TODO - load images
    images = []

    # TODO - expand dims if only one image

    encodings = encoder.encode(images)

    print(encodings.shape)


def train():
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

    # model = tf.keras.applications.mobilenet.MobileNet(
    #     input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001,
    #     include_top=True, weights='imagenet', input_tensor=None, pooling=None,
    #     classes=1000, classifier_activation='softmax'
    # )
    # model.summary()

    # encoder.model.summary()
    # encoder.model.layers[3].summary()

    trn_ds, val_ds = dataset_handler.get_grid_dataset_generators(args.batch_size, args.val_split)
    dataset_size = dataset_handler.get_grid_dataset_size()

    _, best_path = encoder.fit(trn_ds, val_ds, args.epochs, dataset_size)
    encoder.fine_tune(trn_ds, val_ds, args.fine_tune, dataset_size, best_path)


if __name__ == "__main__":
    train()
