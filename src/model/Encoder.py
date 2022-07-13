"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for training of encoder model - encodes an image to a latent vector representing the sports pose.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 26. 06. 2022
"""

from argparse import ArgumentParser
from pathlib import Path
import sys
import time
import datetime
import csv

import tensorflow as tf
from tensorflow.keras import layers

from safe_gpu import safe_gpu

from src.model.DatasetHandler import DatasetHandler, batch_provider


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'dataset',
        type=str,
        help="Location of the directory with dataset.",
    )
    parser.add_argument(
        '-ed', '--encoder_dir',
        type=str,
        default='ckpts/encoder',
        help="Path to directory where encoder checkpoints will be stored.",
    )
    parser.add_argument(
        '-ld', '--log_dir',
        type=str,
        default='logs',
        help="Path to directory where logs will be stored."
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
        '-e', '--fit_epochs',
        type=int,
        default=5,
        help="Number of epochs to be performed on a dataset for fitting."
    )
    parser.add_argument(
        '-f', '--finetune_epochs',
        type=int,
        default=0,
        help="Number of epochs to be performed on a dataset for fine-tuning."
    )
    parser.add_argument(
        '-d', '--encoding_dim',
        type=int,
        default=256,
        help="Dimension of latent space in which an image is represented."
    )
    parser.add_argument(
        '-m', '--margin',
        type=float,
        default=0.01,
        help="Margin used for triplet loss - positive has to be at least by a margin closer to anchor than negative."
    )
    parser.add_argument(
        '-r', '--restore',
        action='store_true',
        help="Use when wanting to restore training from checkpoints."
    )
    parser.add_argument(
        '-S', '--seed',
        type=int,
        default=None,
        help="Seed for dataset shuffling - use to get consistency for training and validation datasets."
    )
    parser.add_argument(
        '-E', '--export_accuracy',
        action='store_true',
        help="Use to turn on exporting of validation accuracy to file logs/accuracies_encoder.csv."
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        default=224,
        help="Dimensions of a training image - height."
    )
    parser.add_argument(
        '-W', '--width',
        type=int,
        default=224,
        help="Dimensions of a training image - width."
    )
    parser.add_argument(
        '-C', '--channels',
        type=int,
        default=3,
        help="Number of channels in used images (e.g. RGB = 3)."
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


def export_accuracies(dimension, margin, seed, val_accuracy, epoch):
    log_file_path = Path("logs/accuracies_encoder.csv")
    if not log_file_path.exists():
        fields = ['Embedding Dimension', 'Margin', 'Seed', 'Validation Accuracy', 'Epoch']
        with open(log_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    fields = [f'{dimension}', f'{margin:.2f}', f'{seed}', f'{val_accuracy:.4f}', f'{epoch}']
    with open(log_file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


@tf.function
def triplet_loss(triplets, margin=0.01):
    a_p_distance = tf.math.reduce_sum(tf.math.square(tf.math.subtract(triplets[:, 0], triplets[:, 1])), axis=1)
    a_n_distance = tf.math.reduce_sum(tf.math.square(tf.math.subtract(triplets[:, 0], triplets[:, 2])), axis=1)

    loss = tf.math.reduce_sum(tf.math.maximum(0., margin + a_p_distance - a_n_distance))
    accuracy = tf.math.reduce_mean(tf.cast(tf.math.less(a_p_distance, a_n_distance), tf.float32))

    return loss, accuracy


class Encoder:
    def __init__(self, height=224, width=224, channels=3, encoding_dim=256, margin=0.01,
                 encoder_dir='ckpts/encoder', restore=False, verbose=False):

        self.encoding_dim = encoding_dim
        self.margin = margin

        self.model = self.optimizer = self.ckpt = self.manager = None
        self.trn_loss = tf.keras.metrics.Mean('trn_loss', dtype=tf.float32)
        self.trn_accuracy = tf.keras.metrics.Mean('trn_accuracy', dtype=tf.float32)
        self.val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        self.val_accuracy = tf.keras.metrics.Mean('val_accuracy', dtype=tf.float32)

        self.train_writer = None
        self.val_writer = None

        self.verbose = verbose
        if self.verbose:
            print("Encoder (En) initialized.")

        self.create_model(height, width, channels, encoding_dim)
        self.set_checkpoints(encoder_dir, restore)

    def create_model(self, height, width, channels, encoding_dim):
        if self.verbose:
            print("En - Creating encoder model.")

        net_input = tf.keras.Input(shape=(height, width, channels))

        # TODO - add "layers.RandomBrightness(0.2)," when running on TF 2.9.
        data_augmentation = tf.keras.Sequential([
            layers.RandomContrast(0.2)
        ])(net_input)

        backbone = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')(data_augmentation)

        head = layers.AveragePooling2D(pool_size=(7, 7))(backbone)
        head = layers.Flatten()(head)
        head = layers.Dense(encoding_dim, activation=None, name='trained')(head)
        # Normalize to a vector on a Unit Hypersphere.
        head = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(head)

        self.model = tf.keras.Model(inputs=net_input, outputs=head, name='encoder')

        self.optimizer = tf.keras.optimizers.Adam()

    def set_checkpoints(self, ckpt_dir, restore):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=100)

        if restore:
            self.ckpt.restore(self.manager.latest_checkpoint)
            self.ckpt.step.assign_add(1)

        if self.verbose and restore and self.manager.latest_checkpoint:
            print(f"En -- Checkpoint restored from {self.manager.latest_checkpoint}")
        elif self.verbose:
            print("En -- Checkpoint initialized in ckpts directory.")

    def set_writers(self, log_dir='logs'):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.create_file_writer(
            str(Path(log_dir) / f'gradient_tape/{current_time}/train')
        )
        self.val_writer = tf.summary.create_file_writer(
            str(Path(log_dir) / f'gradient_tape/{current_time}/val')
        )

    def log_results(self, writer, loss_metrics, acc_metrics, time_metrics, is_train=True):
        loss = loss_metrics.result()
        acc = acc_metrics.result()

        with writer.as_default():
            tf.summary.scalar('loss', loss, step=self.ckpt.step.numpy())
            tf.summary.scalar('accuracy', acc, step=self.ckpt.step.numpy())
            tf.summary.scalar('time', time_metrics, step=self.ckpt.step.numpy())

        if self.verbose:
            print(f"En --- {'Train' if is_train else 'Val'}: "
                  f"Loss = {loss:.6f}, "
                  f"Accuracy = {acc:7.2%}. "
                  f"Time/Tuple = {time_metrics:.2f} ms.")

        loss_metrics.reset_states()
        acc_metrics.reset_states()

        return loss, acc

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            reshaped = tf.reshape(batch,
                                  [batch.shape[0] * batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]])
            batch_encoded = self.model(reshaped, training=True)
            reverted = tf.reshape(batch_encoded, [batch.shape[0], batch.shape[1], self.encoding_dim])
            loss, accuracy = triplet_loss(reverted, self.margin)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        self.trn_loss(loss / len(batch))
        self.trn_accuracy(accuracy)

    def train_on_batches(self, train_ds, sub_dataset_size, batch_size):
        start_time = time.perf_counter()
        for batch in batch_provider(train_ds, batch_size):
            self.train_step(batch)
        end_time = time.perf_counter()
        batch_train_time = (end_time - start_time) / sub_dataset_size * 1e3

        loss, acc = self.log_results(self.train_writer, self.trn_loss, self.trn_accuracy, batch_train_time)

        return loss, acc

    @tf.function
    def val_step(self, batch):
        reshaped = tf.reshape(batch,
                              [batch.shape[0] * batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]])
        batch_encoded = self.model(reshaped, training=False)
        reverted = tf.reshape(batch_encoded, [batch.shape[0], batch.shape[1], self.encoding_dim])
        loss, accuracy = triplet_loss(reverted, self.margin)

        self.val_loss(loss / len(batch))
        self.val_accuracy(accuracy)

    def val_on_batches(self, val_ds, sub_dataset_size, batch_size):
        start_time = time.perf_counter()
        for batch in batch_provider(val_ds, batch_size):
            self.val_step(batch)
        end_time = time.perf_counter()
        batch_val_time = (end_time - start_time) / sub_dataset_size * 1e3

        loss, acc = self.log_results(self.val_writer, self.val_loss, self.val_accuracy, batch_val_time, False)

        return loss, acc

    def fit(self, trn_ds, val_ds, epochs, dataset_size, batch_size):
        if self.verbose:
            print("En - Fitting the model...")

        best_acc = 0
        best_epoch = -1
        best_path = ""

        # FIXME - Check if it does not break the fine-tuning.
        # Freeze backbone (ResNet50).
        self.model.layers[3].trainable = False

        for index in range(epochs):
            if self.verbose:
                print(f"En -- Epoch {self.ckpt.step.numpy() + 1:02d}.")

            self.train_on_batches(trn_ds, dataset_size[0], batch_size)
            _, accuracy = self.val_on_batches(val_ds, dataset_size[1], batch_size)

            save_path = self.manager.save()

            self.ckpt.step.assign_add(1)

            # TODO - Implement as tf.keras.callbacks.History if possible.
            if accuracy > best_acc:
                best_epoch = index
                best_acc = accuracy
                best_path = save_path

            if self.verbose:
                print(f"En --- Checkpoint saved at {save_path}.")

        return best_epoch, best_path, best_acc

    def finetune(self, trn_ds, val_ds, epochs, dataset_size, batch_size, best_path=None):
        if self.verbose:
            print("En - Fine tuning the model...")

        if best_path is not None:
            self.ckpt.restore(best_path)
            self.ckpt.step.assign_add(1)
            if self.verbose:
                print(f"En -- Restored best model from path '{best_path}'.")

        # Unfreeze backbone (ResNet50).
        self.model.layers[3].trainable = True
        self.optimizer = tf.keras.optimizers.Adam(1e-5)

        best_epoch, best_path, best_acc = self.fit(trn_ds, val_ds, epochs, dataset_size, batch_size)

        return best_epoch, best_path, best_acc

    def encode(self, images):
        if self.verbose:
            print("En - Encoding images with the model...")

        if images.ndim == 3:
            images = tf.expand_dims(images, axis=0)
            encoded_images = self.model(images, training=False)
            encoded_images = tf.squeeze(encoded_images, axis=0)
        elif images.ndim == 4:
            encoded_images = self.model(images, training=False)
        else:
            print(f"En -- Dimension of images is incorrect. Has to be 3 (single image) or 4 (multiple images).",
                  file=sys.stderr)
            encoded_images = []

        return encoded_images


def train():
    args = parse_arguments()

    if args.gpu:
        if args.verbose:
            print("Running in GPU enabled mode.")
            print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")
        # noinspection PyUnusedLocal
        gpu_owner = safe_gpu.GPUOwner(placeholder_fn=safe_gpu.tensorflow_placeholder)

    dataset_handler = DatasetHandler(
        args.dataset,
        args.verbose
    )
    encoder = Encoder(
        args.height,
        args.width,
        args.channels,
        args.encoding_dim,
        args.margin,
        args.encoder_dir,
        args.restore,
        args.verbose
    )
    encoder.set_writers(args.log_dir)

    train_ds, val_ds = dataset_handler.get_dataset(args.validation_split, args.seed)
    dataset_size = dataset_handler.get_dataset_size(args.validation_split)

    best_epoch, best_path, best_acc = encoder.fit(train_ds, val_ds, args.fit_epochs, dataset_size, args.batch_size)
    if args.verbose:
        print("En - Training finished.")
        print(f"En - Best accuracy in training was achieved in epoch {best_epoch + 1} and is saved at {best_path}.")

    if args.finetune_epochs > 0:
        best_epoch, best_path, best_acc = encoder.finetune(train_ds, val_ds, args.finetune_epochs,
                                                           dataset_size, args.batch_size, best_path)
        if args.verbose:
            print("En - Fine-tuning finished.")
            print(f"En - Best accuracy in fine-tuning was achieved in "
                  f"epoch {best_epoch + 1} and is saved at {best_path}.")

    if args.export_accuracy:
        export_accuracies(args.encoding_dim, args.margin, args.seed, best_acc, best_epoch + 1)


def info():
    args = parse_arguments()

    encoder = Encoder(
        args.height,
        args.width,
        args.channels,
        args.encoding_dim,
        args.margin,
        args.encoder_dir,
        args.restore,
        args.verbose
    )

    encoder.model.summary()


if __name__ == "__main__":
    train()
    # info()
