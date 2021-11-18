"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for training of encoder model - encodes an image to a latent vector representing the sports pose.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 18. 11. 2021
Source: https://towardsdatascience.com/custom-loss-function-in-tensorflow-2-0-d8fa35405e4e
"""

from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras import layers

from src.model.BatchProvider import BatchProvider
from src.utils.timer import timer


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
        default=128,
        help="Number of triplets in a batch."
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Use to turn on additional text output about what is happening."
    )
    return parser.parse_args()


def triplet_loss(anchor, positive, negative, margin=0.01):
    d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
    d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)
    loss = tf.maximum(0.0, margin + d_pos - d_neg)

    if d_pos < d_neg:
        accuracy = 1
    else:
        accuracy = 0

    return tf.reduce_mean(loss), accuracy


class Encoder:
    def __init__(self, verbose):
        self.verbose = verbose

        self.input_height = 224
        self.input_width = 224
        self.input_channels = 3
        self.output_dimension = 256

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

    def get_gradient(self, anchor, positive, negative):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.model.get_layer(name='trained').variables)
            anchor_encoded = self.model(tf.expand_dims(anchor, axis=0))
            positive_encoded = self.model(tf.expand_dims(positive, axis=0))
            negative_encoded = self.model(tf.expand_dims(negative, axis=0))
            loss, accuracy = triplet_loss(anchor_encoded, positive_encoded, negative_encoded)
            if self.verbose:
                print(f"En --- Loss: {loss:.6f}, Accuracy: {accuracy * 100:6.2f} %")

        gradient = tape.gradient(
            loss,
            [self.model.get_layer(name='trained').variables[0], self.model.get_layer(name='trained').variables[1]]
        )
        # print([var.name for var in tape.watched_variables()])
        return gradient

    def step(self, anchor, positive, negative):
        gradient = self.get_gradient(anchor, positive, negative)
        self.optimizer.apply_gradients(zip(
            gradient,
            [self.model.get_layer(name='trained').variables[0], self.model.get_layer(name='trained').variables[1]]
        ))

    @timer
    def fit(self, batch_provider, batch_size=128, epochs=10):
        if self.verbose:
            print("En - Fitting the model on the training dataset...")

        for epoch in range(epochs):
            if self.verbose:
                print(f"En -- Epoch {epoch:03d}.")

            for batch in batch_provider.batch_generator("train", batch_size):
                for anchor, positive, negative in batch:
                    self.step(anchor, positive, negative)

    @timer
    def evaluate(self, batch_provider, batch_size=128):
        if self.verbose:
            print("En - Evaluating the model on the validation dataset...")
        loss_sum = accuracy_sum = runs = 0

        for batch in batch_provider.batch_generator("val", batch_size):
            for anchor, positive, negative in batch:
                anchor_encoded = self.model(tf.expand_dims(anchor, axis=0))
                positive_encoded = self.model(tf.expand_dims(positive, axis=0))
                negative_encoded = self.model(tf.expand_dims(negative, axis=0))
                loss, accuracy = triplet_loss(anchor_encoded, positive_encoded, negative_encoded)
                runs += 1
                loss_sum += loss
                accuracy_sum += accuracy

                if self.verbose:
                    print(f"En -- Loss: {loss:.6f}, Accuracy: {accuracy * 100:6.2f} %")

        return loss_sum / runs, accuracy_sum / runs


def main():
    args = parse_arguments()
    batch_provider = BatchProvider(args.directory, args.cameras, args.steps, args.width, args.height)
    encoder = Encoder(args.verbose)
    encoder.create_model()

    encoder.fit(batch_provider, args.batch_size, args.epochs)

    loss, accuracy = encoder.evaluate(batch_provider)
    print("En - Overall model evaluation")
    print(f"En -- Loss: {loss:.6f}")
    print(f"En -- Accuracy: {accuracy * 100:6.2f} %")


if __name__ == "__main__":
    main()
