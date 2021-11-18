"""Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for training of encoder model - encodes an image to a latent vector representing the sports pose.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 16. 11. 2021
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
    return parser.parse_args()


def cosine_loss(vector1, vector2, is_positive=True):
    distance = tf.keras.losses.cosine_similarity(vector1, vector2)
    if is_positive:
        loss = distance
    else:
        loss = -distance
    return loss


def triplet_loss(anchor, positive, negative, margin=0.01):
    d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
    d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)
    loss = tf.maximum(0.0, margin + d_pos - d_neg)
    return tf.reduce_mean(loss)


class Encoder:
    def __init__(self):
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
            # loss = cosine_loss(anchor_encoded, positive_encoded, True) \
            #     + cosine_loss(anchor_encoded, negative_encoded, False)
            loss = triplet_loss(anchor_encoded, positive_encoded, negative_encoded)
            print(f"- Loss: {loss}")

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
    def train(self, batch_provider, batch_size=128, epochs=10):
        for epoch in range(epochs):
            print(f"En - Epoch {epoch}.")
            for batch in batch_provider.batch_generator("train", batch_size):
                for anchor, positive, negative in batch:
                    self.step(anchor, positive, negative)

    def inference(self, batch_provider):
        print("Evaluating on new samples.")
        for batch in batch_provider.batch_generator(1):
            anchor, positive, negative = batch[0]
            out_a = self.model(tf.expand_dims(anchor, axis=0))
            out_p = self.model(tf.expand_dims(positive, axis=0))
            out_n = self.model(tf.expand_dims(negative, axis=0))

            print(f"A-P distance: {cosine_loss(out_a, out_p)}")
            print(f"A-N distance: {cosine_loss(out_a, out_n)}")
            print(f"Triplet loss: {triplet_loss(out_a, out_p, out_n)}")

            break


def main():
    args = parse_arguments()

    batch_provider = BatchProvider(args.directory, args.cameras, args.steps, args.width, args.height)

    encoder = Encoder()

    encoder.create_model()
    encoder.train(batch_provider, args.batch_size, args.epochs)
    encoder.inference(batch_provider)


if __name__ == "__main__":
    main()
