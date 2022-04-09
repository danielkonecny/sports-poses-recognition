"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Recognize image from a latent vector.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 09. 04. 2022
"""

from pathlib import Path

import tensorflow as tf

from src.model.Encoder import Encoder
from src.utils.params import parse_arguments


def load_images(directory, height=224, width=224, channels=3):
    image_paths = Path(directory).glob('*.png')
    image_count = len(list(image_paths))
    images = tf.zeros([image_count, height, width, channels])

    for index, image_path in enumerate(image_paths):
        image = tf.io.read_file(image_path)
        images[index] = tf.image.decode_png(image, channels=channels)

    return images


def main():
    args = parse_arguments()
    encoder = Encoder(ckpt_dir=args.ckpt_dir, verbose=args.verbose)

    images = load_images(args.location)
    encoded_images = encoder.encode(images)

    print(encoded_images.shape)


if __name__ == "__main__":
    main()
