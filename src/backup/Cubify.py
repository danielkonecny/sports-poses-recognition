"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Custom layer for preprocessing of input image grids.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 24. 11. 2021
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt

from src.model.DatasetHandler import DatasetHandler
from src.utils.params import parse_arguments


class Cubify(Layer):
    def __init__(self, custom_output):
        super(Cubify, self).__init__()
        self.custom_output = custom_output

    @tf.function
    def call(self, inputs):
        inputs_squeezed = tf.squeeze(inputs, 0)
        repeats = tf.math.floordiv(inputs_squeezed.shape, self.custom_output)
        tmp_shape = tf.stack([repeats, self.custom_output], axis=1)
        tmp_shape_reshaped = tf.reshape(tmp_shape, [-1])
        range_len = tf.range(len(tmp_shape_reshaped))
        order = tf.concat([[range_len[::2], range_len[1::2]]], axis=1)
        order_reshaped = tf.reshape(order, [-1])
        image_grid_reshaped = tf.reshape(inputs_squeezed, tmp_shape_reshaped)
        image_grid_reordered = tf.transpose(image_grid_reshaped, order_reshaped)
        image_grid_finalized = tf.reshape(image_grid_reordered, (-1, *self.custom_output))
        return tf.cast(image_grid_finalized, tf.int64)


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

    trn_ds, _ = dataset_handler.get_grid_dataset_generators(args.batch_size, args.val_split)

    for batch in trn_ds:
        for grid in batch:
            grid = tf.expand_dims(grid, axis=0)

            model = tf.keras.Sequential([
                tf.keras.Input(shape=(args.steps * args.height, args.cameras * args.width, args.channels)),
                Cubify((args.height, args.width, args.channels)),
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.1, fill_mode='nearest')
            ])
            output = model(grid)

            for image in output:
                plt.imshow(image)
                plt.axis("off")
                plt.show()
            break
        break


if __name__ == "__main__":
    test()
