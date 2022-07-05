"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for visualizing of encoded sports poses.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 06. 07. 2022
"""

from pathlib import Path
from contextlib import redirect_stdout, ExitStack
from argparse import ArgumentParser
import json
import csv

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

from src.model.Encoder import Encoder


def parse_arguments():
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="mode",
        help='Choose the mode of recognizer model.'
    )

    parser_tensorboard = subparsers.add_parser(
        'tensorboard',
        help='Create visualization of the embeddings.'
    )
    parser_tensorboard.add_argument(
        'log_dir',
        type=str,
        help="Location of the log directory where visualization data are saved.",
    )
    parser_tensorboard.add_argument(
        'dataset_dir',
        type=str,
        help="Location of the data to be visualized.",
    )
    parser_tensorboard.add_argument(
        'encoder_dir',
        type=str,
        help="Location of the directory with encoder checkpoint.",
    )
    parser_tensorboard.add_argument(
        'encoding_dim',
        type=int,
        help="Dimension of latent space in which an image is represented."
    )

    parser_latex = subparsers.add_parser(
        'latex',
        help='Process visualization export to latex plot.'
    )
    parser_latex.add_argument(
        'json',
        type=str,
        help="Location of the JSON with visualization data.",
    )
    parser_latex.add_argument(
        'labels',
        type=str,
        help="Location of the TSV with metadata about classes.",
    )
    return parser.parse_args()


def load_dataset(directory, batch_size=128):
    print("Vi - Loading dataset...")

    with redirect_stdout(None):
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            batch_size=batch_size,
            image_size=(224, 224),
            shuffle=False
        )

    print(f'Vi -- Number of batches (size {batch_size}) of images loaded: {tf.data.experimental.cardinality(dataset)}.')

    class_names = [x.stem for x in sorted(directory.iterdir()) if x.is_dir()]

    return dataset, class_names


def save_embeddings(dataset, label_names, encoder_dir, log_dir, encoding_dim):
    print("Vi - Saving embeddings with labels...")

    encoder = Encoder(encoder_dir=encoder_dir, encoding_dim=encoding_dim)
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


def tensorboard(args):
    print("Vi - Visualizing...")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    dataset_dir = Path(args.dataset_dir)
    encoder_dir = Path(args.encoder_dir)

    dataset, label_names = load_dataset(dataset_dir)
    save_embeddings(dataset, label_names, encoder_dir, log_dir, args.encoding_dim)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'embedding_metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


def latex(args):
    print("Vi - Exporting to LaTeX...")

    log_dir = Path("logs/latex/t-sne")
    log_dir.mkdir(exist_ok=True, parents=True)

    with open(Path(args.labels)) as label_file:
        labels = label_file.readlines()
        labels = [label.rstrip() for label in labels]
    unique_labels = set(labels)

    with open(Path(args.json)) as json_file:
        points = json.load(json_file)[0]["projections"]

    with ExitStack() as stack:
        csv_files = {
            unique_label: stack.enter_context(open(log_dir / f'{unique_label}.csv', 'w'))
            for unique_label in unique_labels
        }

        header = ["x", "y"]
        writers = {}
        for label, csv_file in csv_files.items():
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writers[label] = writer

        for point, label in zip(points, labels):
            writers[label].writerow([point['tsne-0'], point['tsne-1']])


def main():
    print("Launching Visualizer (Vi)...")

    args = parse_arguments()

    if args.mode == "tensorboard":
        tensorboard(args)
    elif args.mode == "latex":
        latex(args)


if __name__ == "__main__":
    main()
