"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for visualizing of encoded sports poses.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 14. 06. 2022
"""

from pathlib import Path
import contextlib
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

from src.model.Encoder import Encoder


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'log_dir',
        type=str,
        help="Location of the log directory where visualization data are saved.",
    )
    parser.add_argument(
        'dataset_dir',
        type=str,
        help="Location of the data to be visualized.",
    )
    parser.add_argument(
        'encoder_dir',
        type=str,
        help="Location of the directory with encoder checkpoint.",
    )
    return parser.parse_args()


def load_dataset(directory, batch_size=128):
    print("Vi - Loading dataset...")

    with contextlib.redirect_stdout(None):
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            batch_size=batch_size,
            image_size=(224, 224),
            shuffle=False
        )

    print(f'Vi -- Number of batches (size {batch_size}) of images loaded: {tf.data.experimental.cardinality(dataset)}.')

    class_names = [x.stem for x in sorted(directory.iterdir()) if x.is_dir()]

    return dataset, class_names


def save_embeddings(dataset, label_names, encoder_dir, log_dir):
    print("Vi - Saving embeddings with labels...")

    encoder = Encoder(ckpt_dir=encoder_dir)
    embeddings = None

    metadata_path = log_dir / "embedding_metadata.tsv"
    with open(metadata_path, "w") as metadata_file:
        for images, labels in dataset:
            for label in labels:
                metadata_file.write(f"{label_names[label]}\n")
            if embeddings is None:
                embeddings = encoder.encode(images).numpy()
            else:
                embedding = encoder.encode(images).numpy()
                embeddings = np.concatenate((embeddings, embedding))

    embeddings = tf.Variable(embeddings)
    checkpoint = tf.train.Checkpoint(embedding=embeddings)
    checkpoint.save(log_dir / "embedding.ckpt")


def visualize_embeddings(log_dir):
    print("Vi - Visualizing...")

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'embedding_metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


def main():
    print("Launching Visualizer (Vi)...")

    args = parse_arguments()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    dataset_dir = Path(args.dataset_dir)
    encoder_dir = Path(args.encoder_dir)

    dataset, label_names = load_dataset(dataset_dir)
    save_embeddings(dataset, label_names, encoder_dir, log_dir)

    visualize_embeddings(log_dir)


if __name__ == "__main__":
    main()
