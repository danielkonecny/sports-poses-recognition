"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for training of encoder model - encodes an image to a latent vector representing the sports pose.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 21. 11. 2021
"""

from argparse import ArgumentParser
import time

import tensorflow as tf
from tensorflow.keras import layers

from src.model.BatchProvider import BatchProvider, cubify, triplets_in_grid


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
        '-e', '--epochs',
        type=int,
        default=10,
        help="Number of epochs to be performed on a dataset."
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=64,
        help="Number of triplets in a batch."
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Use to turn on additional text output about what is happening."
    )
    return parser.parse_args()


def triplet_loss(anchor, positive, negative, margin=0.01):
    a_p_distance = tf.math.reduce_sum(tf.math.square(anchor - positive))
    a_n_distance = tf.math.reduce_sum(tf.math.square(anchor - negative))
    loss = tf.math.maximum(0.0, margin + a_p_distance - a_n_distance)

    if a_p_distance < a_n_distance:
        accuracy = 1
    else:
        accuracy = 0

    return tf.math.reduce_mean(loss), accuracy


def tuple_loss(nonuplet, triplet_indices):
    loss_sum = accuracy_sum = 0

    for indices in triplet_indices:
        anchor = nonuplet[indices[0]]
        positive = nonuplet[indices[1]]
        negative = nonuplet[indices[2]]
        loss, accuracy = triplet_loss(anchor, positive, negative)

        loss_sum += loss
        accuracy_sum += accuracy

    return loss_sum, accuracy_sum / len(triplet_indices)


class Encoder:
    def __init__(self, verbose):
        self.verbose = verbose

        self.input_height = 224
        self.input_width = 224
        self.input_channels = 3
        self.output_dimension = 256

        self.cameras = 3
        self.steps = 3

        self.train_images, self.test_images, self.train_labels, self.test_labels = None, None, None, None
        self.model = self.optimizer = None

    def create_model(self):
        input_image = tf.keras.Input(shape=(self.input_height, self.input_width, self.input_channels))

        resnet = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                         weights='imagenet',
                                                         input_tensor=input_image)

        head = resnet.output
        head = layers.AveragePooling2D(pool_size=(7, 7))(head)
        head = layers.Flatten()(head)
        head = layers.Dense(self.output_dimension, activation=None, name='trained')(head)
        # Normalize to a vector on a Unit Hypersphere.
        head = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(head)

        self.model = tf.keras.Model(inputs=input_image, outputs=head, name='encoder')
        for layer in self.model.layers:
            layer.trainable = False
        self.model.get_layer(name='trained').trainable = True

        # for layer in self.model.layers:
        #     print(layer.name, layer.trainable)

        self.optimizer = tf.keras.optimizers.Adam()

    def step(self, nonuplet, triplet_indices):
        trained_layers = [
            self.model.get_layer(name='trained').variables[0],
            self.model.get_layer(name='trained').variables[1]
        ]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.model.get_layer(name='trained').variables)
            nonuplet_encoded = self.model(nonuplet)
            loss, accuracy = tuple_loss(nonuplet_encoded, triplet_indices)

        gradient = tape.gradient(loss, trained_layers)
        self.optimizer.apply_gradients(zip(gradient, trained_layers))

        return loss, accuracy

    def fit(self, trn_ds, epochs=10):
        if self.verbose:
            print("En - Fitting the model on the training dataset...")

        triplet_indices = triplets_in_grid((self.steps, self.cameras))

        for epoch in range(epochs):
            if self.verbose:
                print(f"En -- Epoch {epoch:02d}", end=" ")

            loss_sum = accuracy_sum = runs = 0

            start_time = time.perf_counter()

            for batch in trn_ds:
                for grid in batch:
                    grid_reshaped = cubify(grid.numpy(), (self.input_height, self.input_width, self.input_channels))
                    loss, accuracy = self.step(grid_reshaped, triplet_indices)

                    runs += 1
                    loss_sum += loss
                    accuracy_sum += accuracy

            end_time = time.perf_counter()

            if self.verbose:
                print(f"finished after {end_time - start_time:.4f} s"
                      f" - loss={loss_sum / runs:.6f}, accuracy={accuracy_sum / runs:6.2%}.")

    def evaluate(self, val_ds):
        if self.verbose:
            print("En - Evaluating the model on the validation dataset...")

        loss_sum = accuracy_sum = runs = 0

        triplet_indices = triplets_in_grid((self.steps, self.cameras))

        start_time = time.perf_counter()

        for batch in val_ds:
            for grid in batch:
                grid_reshaped = cubify(grid.numpy(), (self.input_height, self.input_width, self.input_channels))
                grid_reshaped_encoded = self.model(grid_reshaped)
                loss, accuracy = tuple_loss(grid_reshaped_encoded, triplet_indices)

                runs += 1
                loss_sum += loss
                accuracy_sum += accuracy

        end_time = time.perf_counter()

        print(f"En -- Finished after {end_time - start_time:.4f} s.")

        return loss_sum / runs, accuracy_sum / runs


def main():
    args = parse_arguments()
    batch_provider = BatchProvider(args.directory, args.cameras, args.steps, args.width, args.height)
    encoder = Encoder(args.verbose)
    encoder.create_model()

    trn_ds, val_ds = batch_provider.get_dataset_generator(args.batch_size)

    encoder.fit(trn_ds, args.epochs)

    loss, accuracy = encoder.evaluate(val_ds)
    print("En - Overall model evaluation")
    print(f"En -- Loss: {loss:.6f}")
    print(f"En -- Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
