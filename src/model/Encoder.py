"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for training of encoder model - encodes an image to a latent vector representing the sports pose.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 21. 11. 2021
"""

import time

import tensorflow as tf
from tensorflow.keras import layers

from src.model.DatasetHandler import DatasetHandler, cubify, triplets_in_grid
from src.utils.params import parse_arguments


def triplet_loss(anchor, positive, negative, margin=0.01):
    a_p_distance = tf.math.reduce_sum(tf.math.square(anchor - positive))
    a_n_distance = tf.math.reduce_sum(tf.math.square(anchor - negative))
    loss = tf.math.maximum(0., margin + a_p_distance - a_n_distance)

    if a_p_distance < a_n_distance:
        accuracy = 1
    else:
        accuracy = 0

    return tf.math.reduce_mean(loss), accuracy


def tuple_loss(n_tuple, triplet_indices, margin=0.01):
    loss_sum = accuracy_sum = 0

    for indices in triplet_indices:
        anchor = n_tuple[indices[0]]
        positive = n_tuple[indices[1]]
        negative = n_tuple[indices[2]]
        loss, accuracy = triplet_loss(anchor, positive, negative, margin)

        loss_sum += loss
        accuracy_sum += accuracy

    return loss_sum, accuracy_sum / len(triplet_indices)


class Encoder:
    def __init__(self, steps, cameras, height, width, channels, encoding_dim, margin, verbose=False):
        self.steps = steps
        self.cameras = cameras
        self.height = height
        self.width = width
        self.channels = channels

        self.encoding_dim = encoding_dim
        self.margin = margin

        self.verbose = verbose

        self.model = self.optimizer = None

        if self.verbose:
            print("Encoder (En) initialized.")

    def create_model(self):
        input_image = tf.keras.Input(shape=(self.height, self.width, self.channels))

        resnet = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                         weights='imagenet',
                                                         input_tensor=input_image)

        head = resnet.output
        head = layers.AveragePooling2D(pool_size=(7, 7))(head)
        head = layers.Flatten()(head)
        head = layers.Dense(self.encoding_dim, activation=None, name='trained')(head)
        # Normalize to a vector on a Unit Hypersphere.
        head = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(head)

        self.model = tf.keras.Model(inputs=input_image, outputs=head, name='encoder')
        for layer in self.model.layers:
            layer.trainable = False
        self.model.get_layer(name='trained').trainable = True

        # for layer in self.model.layers:
        #     print(layer.name, layer.trainable)

        self.optimizer = tf.keras.optimizers.Adam()

    def step(self, n_tuple, triplet_indices):
        trained_layers = [
            self.model.get_layer(name='trained').variables[0],
            self.model.get_layer(name='trained').variables[1]
        ]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.model.get_layer(name='trained').variables)
            n_tuple_encoded = self.model(n_tuple)
            loss, accuracy = tuple_loss(n_tuple_encoded, triplet_indices, self.margin)

        gradient = tape.gradient(loss, trained_layers)
        self.optimizer.apply_gradients(zip(gradient, trained_layers))

        return loss, accuracy

    def fit(self, trn_ds, epochs):
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
                    grid_reshaped = cubify(grid.numpy(), (self.height, self.width, self.channels))
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
        margin = 0.

        start_time = time.perf_counter()

        for batch in val_ds:
            for grid in batch:
                grid_reshaped = cubify(grid.numpy(), (self.height, self.width, self.channels))
                grid_reshaped_encoded = self.model(grid_reshaped)
                loss, accuracy = tuple_loss(grid_reshaped_encoded, triplet_indices, margin)

                runs += 1
                loss_sum += loss
                accuracy_sum += accuracy

        end_time = time.perf_counter()

        print(f"En -- Finished after {end_time - start_time:.4f} s.")

        return loss_sum / runs, accuracy_sum / runs


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
        args.steps,
        args.cameras,
        args.height,
        args.width,
        args.channels,
        args.encoding_dim,
        args.margin,
        args.verbose
    )
    encoder.create_model()

    trn_ds, val_ds = dataset_handler.get_dataset_generators(args.batch_size)

    encoder.fit(trn_ds, args.epochs)

    loss, accuracy = encoder.evaluate(val_ds)
    print("En - Overall model evaluation")
    print(f"En -- Loss: {loss:.6f}")
    print(f"En -- Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
