"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Recognize image from a latent vector with a dense neural network.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 19. 04. 2022
"""

from pathlib import Path
import sys
import contextlib

import tensorflow as tf
from tensorflow.keras import layers

from src.model.Encoder import Encoder
from src.utils.params import parse_arguments


class RecognizerNN:
    def __init__(self, directory, ckpt_encoder_dir, ckpt_recognizer_dir, verbose, height=224, width=224, channels=3):
        self.class_names = [x.stem for x in sorted(Path(directory).iterdir()) if x.is_dir()]
        ckpt_recognizer_dir = Path(ckpt_recognizer_dir) / "ckpt"

        self.classifier = tf.keras.Sequential([
            layers.InputLayer(input_shape=(height, width, channels)),
            Encoder(ckpt_dir=ckpt_encoder_dir, verbose=verbose).model,
            layers.Dense(128, activation=layers.LeakyReLU(alpha=0.01)),
            layers.Dense(len(self.class_names), activation=layers.Softmax())
        ], name='recognizer')
        self.classifier.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss=tf.keras.losses.CategoricalCrossentropy(),
                                metrics=['accuracy'])
        self.ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_recognizer_dir,
                                                                save_weights_only=True,
                                                                verbose=1)

        self.verbose = verbose
        if self.verbose:
            print("Recognizer - Dense Neural Network (RD) initialized.")

    def load_dataset(self, directory, batch_size=16):
        if self.verbose:
            print("RD - Loading dataset...")

        directory = Path(directory)

        random_seed = tf.random.uniform(shape=(), minval=1, maxval=2 ** 32, dtype=tf.int64)

        with contextlib.redirect_stdout(None):
            train_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                label_mode='categorical',
                batch_size=batch_size,
                image_size=(224, 224),
                seed=random_seed,
                validation_split=0.2,
                subset="training"
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                label_mode='categorical',
                batch_size=batch_size,
                image_size=(224, 224),
                seed=random_seed,
                validation_split=0.2,
                subset="validation"
            )

        if self.verbose:
            print(f'RD -- Number of train batches (size {batch_size}) loaded: '
                  f'{tf.data.experimental.cardinality(train_ds)}.')
            print(f'RD -- Number of validation batches (size {batch_size}) loaded: '
                  f'{tf.data.experimental.cardinality(val_ds)}.')

        return train_ds, val_ds

    def fit(self, train_ds, val_ds, epochs=5):
        if self.verbose:
            print("RD - Fitting the model...")

        self.classifier.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[self.ckpt_callback]
        )

    def evaluate(self, val_ds):
        for batch_image, batch_label in val_ds:
            print(f"Evaluated: {self.classifier(batch_image)} - Original: {batch_label}")

        self.classifier.evaluate(val_ds)

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

    def save(self):
        pass

    def load(self):
        pass


def train():
    args = parse_arguments()

    recognizer_neighbors = RecognizerNN(args.location, args.ckpt_encoder, args.ckpt_recognizer, args.verbose)
    train_ds, val_ds = recognizer_neighbors.load_dataset(args.location)
    recognizer_neighbors.fit(train_ds, val_ds)
    # recognizer_neighbors.save()
    # recognizer_neighbors.load("ckpts/best/recognizer_neighbors.pkl")
    recognizer_neighbors.evaluate(val_ds)


if __name__ == "__main__":
    train()
