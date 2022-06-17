"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Recognize image from a latent vector with a dense neural network.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 17. 06. 2022
"""

from argparse import ArgumentParser
from pathlib import Path
import sys
import contextlib

import tensorflow as tf
from tensorflow.keras import layers

from safe_gpu import safe_gpu

from src.model.Encoder import Encoder


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'dataset',
        type=str,
        help="Location of the directory with dataset.",
    )
    parser.add_argument(
        'encoder_dir',
        type=str,
        help="Location of the directory with encoder checkpoint.",
    )
    parser.add_argument(
        '--recognizer_dir',
        type=str,
        default='ckpts_recognizer',
        help="Location of the directory where recognizer checkpoints will be stored."
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=16,
        help="Batch size for training."
    )
    parser.add_argument(
        '-s', '--validation_split',
        type=float,
        default=0.2,
        help="Number between 0 and 1 representing proportion of dataset to be used for validation."
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=5,
        help="Number of epochs to be performed on a dataset for fitting."
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Use to turn on additional text output about what is happening."
    )
    parser.add_argument(
        '-g', '--gpu',
        action='store_true',
        help="Use to turn on Safe GPU command to run on a machine with multiple GPUs."
    )
    return parser.parse_args()


class Recognizer:
    def __init__(self, directory, encoder_dir, recognizer_dir, verbose, height=224, width=224, channels=3):
        self.class_names = [x.stem for x in sorted(Path(directory).iterdir()) if x.is_dir()]

        self.classifier = tf.keras.Sequential([
            layers.InputLayer(input_shape=(height, width, channels)),
            Encoder(encoder_dir=encoder_dir, verbose=verbose).model,
            layers.Dense(64, activation=layers.LeakyReLU(alpha=0.01)),
            layers.Dense(len(self.class_names), activation=layers.Softmax())
        ], name='recognizer')

        self.classifier.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=tf.keras.losses.CategoricalCrossentropy(),
                                metrics=['accuracy'])

        self.ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=(Path(recognizer_dir) / "ckpt-epoch{epoch:02d}-val_acc{val_accuracy:.2f}"),
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )

        self.verbose = verbose
        if self.verbose:
            print("Recognizer - Dense Neural Network (RD) initialized.")

    def load_dataset(self, directory, batch_size, validation_split, height=224, width=224):
        if self.verbose:
            print("RD - Loading dataset...")

        directory = Path(directory)

        random_seed = tf.random.uniform(shape=(), minval=1, maxval=2 ** 32, dtype=tf.int64)

        with contextlib.redirect_stdout(None):
            train_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                label_mode='categorical',
                batch_size=batch_size,
                image_size=(height, width),
                seed=random_seed,
                validation_split=validation_split,
                subset="training"
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                label_mode='categorical',
                batch_size=batch_size,
                image_size=(height, width),
                seed=random_seed,
                validation_split=validation_split,
                subset="validation"
            )

        if self.verbose:
            print(f'RD -- Number of train batches (size {batch_size}) loaded: '
                  f'{tf.data.experimental.cardinality(train_ds)}.')
            print(f'RD -- Number of validation batches (size {batch_size}) loaded: '
                  f'{tf.data.experimental.cardinality(val_ds)}.')

        return train_ds, val_ds

    def evaluate_samples(self, val_ds):
        for batch_image, batch_label in val_ds:
            for image, label in zip(batch_image, batch_label):
                result = self.classifier(tf.expand_dims(image, 0))[0]

                truth_class_index = tf.math.argmax(label)
                class_index = tf.math.argmax(result)

                if class_index == truth_class_index:
                    print(f"CORRECT: {self.class_names[class_index]:>9} = {result[class_index]:6.2%}")
                else:
                    print(f"FALSE:   {self.class_names[truth_class_index]:>9} = {result[truth_class_index]:6.2%} "
                          f"not {self.class_names[class_index]:>9} = {result[class_index]:6.2%}")

    def predict(self, images):
        if self.verbose:
            print("En - Encoding images with the model...")

        if images.ndim == 3:
            images = tf.expand_dims(images, axis=0)
            predictions = self.classifier(images, training=False)
            predictions = tf.squeeze(predictions, axis=0)
        elif images.ndim == 4:
            predictions = self.classifier(images, training=False)
        else:
            print(f"En -- Dimension of images is incorrect. Has to be 3 (single image) or 4 (multiple images).",
                  file=sys.stderr)
            predictions = []

        return predictions


def train():
    args = parse_arguments()

    if args.gpu:
        if args.verbose:
            print("Running in GPU enabled mode.")
            print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")
        # noinspection PyUnusedLocal
        gpu_owner = safe_gpu.GPUOwner(placeholder_fn=safe_gpu.tensorflow_placeholder)

    recognizer = Recognizer(args.dataset, args.encoder_dir, args.recognizer_dir, args.verbose)
    train_ds, val_ds = recognizer.load_dataset(args.dataset, args.batch_size, args.validation_split)

    recognizer.classifier.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=[recognizer.ckpt_callback]
    )

    recognizer.evaluate_samples(val_ds)


if __name__ == "__main__":
    train()
