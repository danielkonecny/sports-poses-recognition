"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for visualizing of encoded sports poses.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 13. 06. 2022
"""

from pathlib import Path
import contextlib

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

from src.model.Encoder import Encoder


def load_dataset(directory, batch_size=128):
    print("Vi - Loading dataset...")

    with contextlib.redirect_stdout(None):
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            label_mode='int',
            batch_size=batch_size,
            image_size=(224, 224),
            shuffle=False
        )

    print(f'Vi -- Number of images loaded: {tf.data.experimental.cardinality(dataset)}.')

    label_names = []
    for subdir_path in directory.iterdir():
        if subdir_path.is_dir():
            label_names.append(subdir_path.stem)

    return dataset, label_names


def save_embeddings(dataset, label_names, ckpt_encoder_dir, log_dir):
    print("Vi - Saving embeddings with labels...")

    encoder = Encoder(ckpt_dir=ckpt_encoder_dir)
    embeddings = None

    metadata_path = log_dir / "metadata.tsv"
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
    checkpoint.save(log_dir / "embeddings.ckpt")


def visualize_embeddings(log_dir):
    print("Vi - Visualizing...")

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


def main():
    print("Launching Visualizer (Vi)...")

    dataset_dir = Path("data/left")
    ckpt_encoder_dir = Path("ckpts/best")
    log_dir = Path("logs/visualize")
    log_dir.mkdir(exist_ok=True, parents=True)

    dataset, label_names = load_dataset(dataset_dir)
    save_embeddings(dataset, label_names, ckpt_encoder_dir, log_dir)

    visualize_embeddings(log_dir)


if __name__ == "__main__":
    main()
