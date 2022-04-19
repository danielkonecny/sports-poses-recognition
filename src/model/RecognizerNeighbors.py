"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Recognize image from a latent vector with k-nearest neighbors method.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 16. 04. 2022
"""

from pathlib import Path
import contextlib
import pickle

import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier

from src.model.Encoder import Encoder
from src.utils.params import parse_arguments


def convert_dataset(dataset):
    images, labels = zip(*dataset)

    images = tf.squeeze(tf.convert_to_tensor(images), axis=1)
    labels = tf.squeeze(tf.convert_to_tensor(labels), axis=1)

    return images, labels


class RecognizerNeighbors:
    def __init__(self, directory, neighbors, ckpt_encoder_dir, ckpt_recognizer_dir, verbose):
        ckpt_encoder_dir = Path(ckpt_encoder_dir)
        self.ckpt_recognizer_dir = Path(ckpt_recognizer_dir) / "recognizer_neighbors.pkl"

        self.class_names = [x.stem for x in sorted(directory.iterdir()) if x.is_dir()]

        self.encoder = Encoder(ckpt_dir=ckpt_encoder_dir, verbose=verbose)
        self.classifier = KNeighborsClassifier(n_neighbors=neighbors)

        self.verbose = verbose
        if self.verbose:
            print("Recognizer - Neighbors (RN) initialized.")

    def load_dataset(self, directory):
        if self.verbose:
            print("RN - Loading dataset...")

        directory = Path(directory)

        random_seed = tf.random.uniform(shape=(), minval=1, maxval=2 ** 32, dtype=tf.int64)

        with contextlib.redirect_stdout(None):
            train_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                label_mode='int',
                batch_size=1,
                image_size=(224, 224),
                seed=random_seed,
                validation_split=0.2,
                subset="training"
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                label_mode='int',
                batch_size=1,
                image_size=(224, 224),
                seed=random_seed,
                validation_split=0.2,
                subset="validation"
            )

        if self.verbose:
            print(f'RN -- Number of train images loaded: {tf.data.experimental.cardinality(train_ds)}.')
            print(f'RN -- Number of validation images loaded: {tf.data.experimental.cardinality(val_ds)}.')

        train_images, train_labels = convert_dataset(train_ds)
        val_images, val_labels = convert_dataset(val_ds)

        return train_images, train_labels, val_images, val_labels

    def fit(self, train_images, train_labels):
        if self.verbose:
            print("RN - Fitting the model...")

        train_encodings = self.encoder.encode(train_images)

        self.classifier.fit(train_encodings, train_labels)

    def evaluate(self, val_images, val_labels):
        if self.verbose:
            print("RN - Evaluating the model...")

        val_encodings = self.encoder.encode(val_images)

        predictions = self.classifier.predict(val_encodings)
        probabilities = self.classifier.predict_proba(val_encodings)
        accuracy = self.classifier.score(val_encodings, val_labels)

        if self.verbose:
            print(f"RN -- Analyzing individual samples...")
            for prediction, truth, probability in zip(predictions, val_labels, probabilities):
                if prediction == truth:
                    print(f"RN --- CORRECT: Guessed {self.class_names[prediction]} correctly "
                          f"with probability {probability[prediction]:.2%}.")
                else:
                    print(f"RN --- WRONG: Guessed {self.class_names[prediction]} with probability "
                          f"{probability[prediction]:.2%} but the result is {self.class_names[truth]} "
                          f"which had probability {probability[truth]:.2%}.")

        print(f"RN -- Accuracy of the model is {accuracy:.2%}.")

    def save(self):
        pickle.dump(self.classifier, open(self.ckpt_recognizer_dir, 'wb'))

        if self.verbose:
            print("RN - Model saved.")

    def load(self):
        self.classifier = pickle.load(open(self.ckpt_recognizer_dir, 'rb'))

        if self.verbose:
            print("RN - Model loaded.")


def train():
    args = parse_arguments()

    recognizer_neighbors = RecognizerNeighbors(args.location, args.neighbors,
                                               args.ckpt_encoder, args.ckpt_recognizer, args.verbose)

    train_images, train_labels, val_images, val_labels = recognizer_neighbors.load_dataset(args.location)
    recognizer_neighbors.fit(train_images, train_labels)
    recognizer_neighbors.save()
    recognizer_neighbors.load()
    recognizer_neighbors.evaluate(val_images, val_labels)


if __name__ == "__main__":
    train()
