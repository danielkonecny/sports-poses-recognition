"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for training of encoder model - encodes an image to a latent vector representing the sports pose.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 08. 12. 2021
"""

from pathlib import Path
import time
import datetime

import tensorflow as tf
from tensorflow.keras import layers

from src.model.Cubify import Cubify
from src.model.DatasetHandler import DatasetHandler, triplets_in_grid
from src.utils.params import parse_arguments


@tf.function
def triplet_loss(anchor, positive, negative, margin=0.01):
    a_p_distance = tf.math.reduce_sum(tf.math.square(anchor - positive))
    a_n_distance = tf.math.reduce_sum(tf.math.square(anchor - negative))
    loss = tf.math.maximum(0., margin + a_p_distance - a_n_distance)

    if a_p_distance < a_n_distance:
        accuracy = 1.
    else:
        accuracy = 0.

    return tf.math.reduce_mean(loss), accuracy


@tf.function
def tuple_loss(n_tuple, triplet_indices, margin=0.01):
    loss_sum = accuracy_sum = 0.

    for indices in triplet_indices:
        anchor = n_tuple[indices[0]]
        positive = n_tuple[indices[1]]
        negative = n_tuple[indices[2]]
        loss, accuracy = triplet_loss(anchor, positive, negative, margin)

        loss_sum += loss
        accuracy_sum += accuracy

    return loss_sum, accuracy_sum / len(triplet_indices)


class Encoder:
    def __init__(self, directory, steps, cameras, height, width, channels, encoding_dim, margin,
                 ckpt_dir, restore, verbose=False):
        self.directory = Path(directory)

        self.steps = steps
        self.cameras = cameras
        self.height = height
        self.width = width
        self.channels = channels

        self.encoding_dim = encoding_dim
        self.margin = margin

        self.ckpt_dir = str(self.directory / ckpt_dir)
        self.restore = restore

        self.verbose = verbose

        self.model = self.optimizer = self.ckpt = self.manager = None
        self.trn_loss = tf.keras.metrics.Mean('trn_loss', dtype=tf.float32)
        self.trn_accuracy = tf.keras.metrics.Mean('trn_accuracy', dtype=tf.float32)
        self.val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        self.val_accuracy = tf.keras.metrics.Mean('val_accuracy', dtype=tf.float32)

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.create_file_writer(
            str(self.directory / f'logs/{self.current_time}/gradient_tape/train')
        )
        self.val_writer = tf.summary.create_file_writer(
            str(self.directory / f'logs/{self.current_time}/gradient_tape/val')
        )

        if self.verbose:
            print("Encoder (En) initialized.")

    def create_model(self):
        if self.verbose:
            print("En - Creating encoder model.")

        input_image = tf.keras.Input(shape=(self.height * self.steps, self.width * self.cameras, self.channels))

        "'__call__' method is inherited from 'tf.keras.layers.Layer' and calls defined 'call' method, so no problem."
        # noinspection PyCallingNonCallable
        cubify = Cubify((self.height, self.width, self.channels))(input_image)

        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1, fill_mode='nearest'),
        ])(cubify)

        resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')(data_augmentation)

        head = layers.AveragePooling2D(pool_size=(7, 7))(resnet)
        head = layers.Flatten()(head)
        head = layers.Dense(self.encoding_dim, activation=None, name='trained')(head)
        # Normalize to a vector on a Unit Hypersphere.
        head = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(head)

        self.model = tf.keras.Model(inputs=input_image, outputs=head, name='encoder')
        for layer in self.model.layers:
            layer.trainable = False
        self.model.get_layer(name='trained').trainable = True

        self.optimizer = tf.keras.optimizers.Adam()

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=5)

        if self.restore:
            self.ckpt.restore(self.manager.latest_checkpoint)

        if self.verbose and self.restore and self.manager.latest_checkpoint:
            print(f"En -- Checkpoint restored from {self.manager.latest_checkpoint}")
        elif self.verbose:
            print("En -- Checkpoint initialized in ckpts directory.")

    @tf.function
    def train_step(self, grid, triplet_indices):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.model.get_layer(name='trained').variables)
            n_tuple = tf.expand_dims(grid, axis=0)
            n_tuple_encoded = self.model(n_tuple, training=True)
            loss, accuracy = tuple_loss(n_tuple_encoded, triplet_indices, self.margin)

        trained_layers = [
            self.model.get_layer(name='trained').kernel,
            self.model.get_layer(name='trained').bias
        ]
        gradient = tape.gradient(loss, trained_layers)
        self.optimizer.apply_gradients(zip(gradient, trained_layers))

        self.trn_loss(loss / len(triplet_indices))
        self.trn_accuracy(accuracy)

    def train_on_batches(self, trn_ds, triplet_indices, subdataset_size):
        start_time = time.perf_counter()
        for batch in trn_ds:
            for grid in batch:
                self.train_step(grid, triplet_indices)
        end_time = time.perf_counter()
        tuple_train_time = (end_time - start_time) / subdataset_size * 1e3

        with self.train_writer.as_default():
            tf.summary.scalar('loss', self.trn_loss.result(), step=self.ckpt.step.numpy())
            tf.summary.scalar('accuracy', self.trn_accuracy.result(), step=self.ckpt.step.numpy())
            tf.summary.scalar('time', tuple_train_time, step=self.ckpt.step.numpy())

        if self.verbose:
            print(f"En --- Train: Loss = {self.trn_loss.result():.6f},"
                  f" Accuracy = {self.trn_accuracy.result():7.2%}."
                  f" Time / Tuple = {tuple_train_time:.2f} ms.")

        self.trn_loss.reset_states()
        self.trn_accuracy.reset_states()

    @tf.function
    def val_step(self, grid, triplet_indices):
        margin = self.margin

        n_tuple = tf.expand_dims(grid, axis=0)
        n_tuple_encoded = self.model(n_tuple, training=False)
        loss, accuracy = tuple_loss(n_tuple_encoded, triplet_indices, margin)

        self.val_loss(loss / len(triplet_indices))
        self.val_accuracy(accuracy)

    def val_on_batches(self, val_ds, triplet_indices, subdataset_size):
        start_time = time.perf_counter()
        for batch in val_ds:
            for grid in batch:
                self.val_step(grid, triplet_indices)
        end_time = time.perf_counter()
        tuple_train_time = (end_time - start_time) / subdataset_size * 1e3

        with self.val_writer.as_default():
            tf.summary.scalar('loss', self.val_loss.result(), step=self.ckpt.step.numpy())
            tf.summary.scalar('accuracy', self.val_accuracy.result(), step=self.ckpt.step.numpy())
            tf.summary.scalar('time', tuple_train_time, step=self.ckpt.step.numpy())

        if self.verbose:
            print(f"En --- Validation: Loss = {self.val_loss.result():.6f},"
                  f" Accuracy = {self.val_accuracy.result():7.2%},"
                  f" Time / Tuple = {tuple_train_time:.2f} ms.")

        self.val_loss.reset_states()
        self.val_accuracy.reset_states()

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
        args.directory,
        args.steps,
        args.cameras,
        args.height,
        args.width,
        args.verbose
    )
    encoder = Encoder(
        args.directory,
        args.steps,
        args.cameras,
        args.height,
        args.width,
        args.channels,
        args.encoding_dim,
        args.margin,
        args.ckpt_dir,
        args.restore,
        args.verbose
    )
    encoder.create_model()

    trn_ds, val_ds = dataset_handler.get_dataset_generators(args.batch_size, args.val_split)
    dataset_size = dataset_handler.get_dataset_size()

    encoder.fit(trn_ds, val_ds, args.epochs, dataset_size)


if __name__ == "__main__":
    main()
