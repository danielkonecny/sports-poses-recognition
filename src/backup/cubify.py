"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Cubify as a function instead of layer that is used.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 24. 11. 2021
"""

import tensorflow as tf


def cubify(image_grid, new_shape):
    """
    Converts grid of images from one image to 1-D array of all images.
    new_shape must divide old_shape evenly or else ValueError will be raised.
    Source: https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes/42298440#42298440
    """
    repeats = tf.math.floordiv(image_grid.shape, new_shape)
    tmp_shape = tf.stack([repeats, new_shape], axis=1)
    tmp_shape_reshaped = tf.reshape(tmp_shape, [-1])
    range_len = tf.range(len(tmp_shape_reshaped))
    order = tf.concat([[range_len[::2], range_len[1::2]]], axis=1)
    order_reshaped = tf.reshape(order, [-1])
    image_grid_reshaped = tf.reshape(image_grid, tmp_shape_reshaped)
    image_grid_reordered = tf.transpose(image_grid_reshaped, order_reshaped)
    image_grid_finalized = tf.reshape(image_grid_reordered, (-1, *new_shape))
    return tf.cast(image_grid_finalized, tf.int64)
