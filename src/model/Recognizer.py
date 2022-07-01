"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Recognize image from a latent vector with a dense neural network.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 21. 06. 2022
"""

from argparse import ArgumentParser
from pathlib import Path
import logging
import contextlib

import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

from safe_gpu import safe_gpu

from src.model.Encoder import Encoder


def parse_arguments():
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="mode",
        help='Choose the mode of recognizer model.'
    )

    parser_fit = subparsers.add_parser(
        'fit',
        help='Options when fitting the recognizer model.'
    )
    parser_fit.add_argument(
        'dataset',
        type=str,
        help="Location of the directory with dataset."
    )
    parser_fit.add_argument(
        'encoder_dir',
        type=str,
        help="Location of the directory with encoder checkpoint."
    )
    parser_fit.add_argument(
        '--recognizer_dir',
        type=str,
        default='models/recognizer',
        help="Location of the directory where model will be stored."
    )
    parser_fit.add_argument(
        '--ckpt_dir',
        type=str,
        default='ckpts/recognizer',
        help="Location of the directory where model checkpoints will be stored."
    )
    parser_fit.add_argument(
        '-b', '--batch_size',
        type=int,
        default=16,
        help="Batch size for training."
    )
    parser_fit.add_argument(
        '-s', '--validation_split',
        type=float,
        default=.2,
        help="Number between 0 and 1 representing proportion of dataset to be used for validation."
    )
    parser_fit.add_argument(
        '-e', '--epochs',
        type=int,
        default=5,
        help="Number of epochs to be performed on a dataset for fitting."
    )
    parser_fit.add_argument(
        '-p', '--dataset_portion',
        type=float,
        default=1.,
        help="Portion of dataset that is used, number between 0 and 1 (0 not included)."
    )
    parser_fit.add_argument(
        '-S', '--seed',
        type=int,
        default=None,
        help="Seed for dataset shuffling - use to get consistency for training and validation datasets."
    )
    parser_fit.add_argument(
        '-E', '--export_accuracy',
        action='store_true',
        help="Use to turn on exporting of validation accuracy to file logs/accuracies.csv."
    )
    parser_fit.add_argument(
        '-t', '--top_k_accuracy',
        nargs="+",
        type=int,
        default=[1],
        help="Evaluate top K accuracy of the model, allows for multiple Ks."
    )
    parser_fit.add_argument(
        '-g', '--gpu',
        action='store_true',
        help="Use to turn on Safe GPU command to run on a machine with multiple GPUs."
    )
    parser_fit.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Use to turn on additional text output about what is happening."
    )

    parser_predict = subparsers.add_parser(
        'predict',
        help='Options when predicting from the recognizer model.'
    )
    parser_predict.add_argument(
        'images',
        type=str,
        help="Location of the directory with images to predict."
    )
    parser_predict.add_argument(
        'recognizer',
        type=str,
        help="Location of recognizer model as h5 file."
    )
    parser_predict.add_argument(
        'labels',
        type=str,
        help="Location of labels saved in tsv file."
    )
    parser_predict.add_argument(
        '-g', '--gpu',
        action='store_true',
        help="Use to turn on Safe GPU command to run on a machine with multiple GPUs."
    )
    parser_predict.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Use to turn on additional text output about what is happening."
    )

    parser_info = subparsers.add_parser(
        'info',
        help='Options when getting info about recognizer model.'
    )
    parser_info.add_argument(
        'recognizer',
        type=str,
        help="Location of recognizer model as h5 file."
    )
    parser_info.add_argument(
        '-g', '--gpu',
        action='store_true',
        help="Use to turn on Safe GPU command to run on a machine with multiple GPUs."
    )
    parser_info.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Use to turn on additional text output about what is happening."
    )
    return parser.parse_args()


class Recognizer:
    def __init__(self, model):
        self.model = model

        logging.info("Recognizer (Re) initialized.")

    @classmethod
    def create(cls, encoder_dir, label_count, top_ks=None):
        encoder = Encoder(encoder_dir=encoder_dir).model
        # TODO - Causes CustomMaskWarning in TF 2.7, will disappear when upgraded to higher version.
        config = encoder.get_config()

        model = tf.keras.Sequential(
            layers=[
                layers.InputLayer(input_shape=config["layers"][0]["config"]["batch_input_shape"][1:]),
                encoder,
                layers.Dense(64, activation=layers.LeakyReLU(alpha=0.01)),
                layers.Dense(label_count, activation=layers.Softmax())
            ],
            name='model'
        )

        top_k_accuracies = []
        if top_ks is None:
            top_ks = [1]
        for top_k in top_ks:
            top_k_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=top_k, name=f'top_{top_k}_categorical_accuracy')
            top_k_accuracies.append(top_k_accuracy)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=top_k_accuracies
        )

        return cls(model)

    @classmethod
    def load(cls, recognizer_dir):
        model = tf.keras.models.load_model(recognizer_dir)

        return cls(model)

    @staticmethod
    def load_dataset(directory, batch_size, validation_split, dataset_portion=1., seed=None, height=224, width=224):
        logging.info("Re - Loading dataset...")

        directory = Path(directory)

        if seed is None:
            random_seed = tf.random.uniform(shape=(), minval=1, maxval=2 ** 32, dtype=tf.int64)
        else:
            random_seed = seed

        with contextlib.redirect_stdout(None):
            # TODO - TF 2.10 enables to return train and val at once as a tuple, shorten then.
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
        # FIXME - Sharding causes validation TF.data.Dataset to somehow not be consistent.
        if dataset_portion < 1.:
            train_ds = train_ds.shard(num_shards=tf.cast(1 / dataset_portion, tf.int64), index=0)
            val_ds = val_ds.shard(num_shards=tf.cast(1 / dataset_portion, tf.int64), index=0)

        logging.info(f'Re -- Number of train batches (size {batch_size}) loaded: '
                     f'{tf.data.experimental.cardinality(train_ds)}.')
        logging.info(f'Re -- Number of validation batches (size {batch_size}) loaded: '
                     f'{tf.data.experimental.cardinality(val_ds)}.')

        return train_ds, val_ds

    def load_images(self, path):
        logging.info("Re - Loading images...")

        path = Path(path)

        # TODO - Causes CustomMaskWarning in TF 2.7, will disappear when upgraded to higher version.
        config = self.model.get_config()

        with contextlib.redirect_stdout(None):
            images = tf.keras.utils.image_dataset_from_directory(
                path,
                labels=None,
                label_mode=None,
                batch_size=1,
                image_size=config["layers"][0]["config"]["batch_input_shape"][1:3],
                shuffle=False
            )

        logging.info(f'Re -- Number of images loaded: {tf.data.experimental.cardinality(images)}.')

        return images

    def evaluate_samples(self, val_ds, labels):
        for batch_image, batch_label in val_ds:
            results = self.model(batch_image)

            for result, label in zip(results, batch_label):
                truth_class_index = tf.math.argmax(label)
                class_index = tf.math.argmax(result)

                if class_index == truth_class_index:
                    print(f"CORRECT: {labels[class_index]:>9} = {result[class_index]:6.2%}")
                else:
                    print(f"FALSE:   {labels[truth_class_index]:>9} = {result[truth_class_index]:6.2%} "
                          f"not {labels[class_index]:>9} = {result[class_index]:6.2%}")

    def predict(self, images):
        logging.info("Re - Encoding images with the model...")

        predictions = []

        for batch in images:
            prediction = self.model(batch, training=False)
            predictions.append(prediction)

        return predictions

    @staticmethod
    def export_accuracies(fields):
        log_file_path = Path("logs/accuracies.csv")
        field_names = list(fields.keys())

        if log_file_path.exists():
            accuracy_df = pd.read_csv(log_file_path)
            for field_name in field_names:
                if field_name not in accuracy_df.columns:
                    logging.error("Re - Accuracy logging file header is not corresponding. "
                                  "Saving the file as .bak and creating a new one.")
                    accuracy_df.to_csv(log_file_path.parent / (log_file_path.stem + '.bak.csv'), index=False)
                    accuracy_df = pd.DataFrame(columns=field_names)
        else:
            accuracy_df = pd.DataFrame(columns=field_names)

        new_row = pd.DataFrame(fields, index=[0])
        accuracy_df = pd.concat([accuracy_df, new_row])
        accuracy_df.to_csv(log_file_path, index=False)


def fit(args):
    labels = [x.stem for x in sorted(Path(args.dataset).iterdir()) if x.is_dir()]

    recognizer = Recognizer.create(args.encoder_dir, len(labels), args.top_k_accuracy)
    train_ds, val_ds = recognizer.load_dataset(args.dataset, args.batch_size, args.validation_split,
                                               args.dataset_portion, args.seed)

    logging.info("Re - Fitting the model...")
    history = recognizer.model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=(Path(args.ckpt_dir) / "ckpt-epoch{epoch:02d}"),
                save_weights_only=True,
                monitor=f'val_top_{args.top_k_accuracy[0]}_categorical_accuracy',
                mode='max',
                verbose=1
            )
        ]
    )

    # Restore weights of model that had the highest accuracy on validation dataset.
    epoch = tf.math.argmax(history.history[f'val_top_{args.top_k_accuracy[0]}_categorical_accuracy']) + 1
    val_accuracy = tf.math.reduce_max(history.history[f'val_top_{args.top_k_accuracy[0]}_categorical_accuracy'])
    recognizer.model.load_weights(Path(args.ckpt_dir) / f"ckpt-epoch{epoch:02d}")
    logging.info(f"Re - Restored checkpoint from epoch {epoch:02d} with validation accuracy {val_accuracy:.6f}.")

    if args.export_accuracy:
        fields = {
            'Model': 'Self-Supervised',
            'Training Data Portion': f'{1 - args.validation_split}',
            'Seed': f'{args.seed}'
        }
        for k in args.top_k_accuracy:
            fields[f'Top-{k} Validation Accuracy'] = \
                f'{history.history[f"val_top_{k}_categorical_accuracy"][epoch - 1]:.4f}'
        recognizer.export_accuracies(fields)

    logging.info("Re - Evaluating the model...")
    recognizer.model.evaluate(val_ds)
    recognizer.evaluate_samples(val_ds, labels)

    logging.info("Re - Saving the model...")
    # TODO - Change to this when upgraded above TF 2.7: recognizer.model.save(args.recognizer_dir)
    recognizer.model.save(Path(args.recognizer_dir) / "recognizer.h5", save_format='h5')

    label_path = Path(args.recognizer_dir) / "labels.tsv"
    with open(label_path, "w") as label_file:
        for label in labels:
            label_file.write(f"{label}\n")


def predict(args):
    logging.info("Re - Predicting images with recognizer...")

    with open(Path(args.labels)) as label_file:
        labels = label_file.readlines()
        labels = [label.rstrip() for label in labels]

    recognizer = Recognizer.load(args.recognizer)
    images = recognizer.load_images(args.images)

    predictions = recognizer.predict(images)

    for prediction in predictions:
        label_index = tf.math.argmax(prediction[0])
        logging.info(f"Label {labels[label_index]:>9} with {prediction[0][label_index]:6.2%} certainty.")
        print(labels[label_index])


def info(args):
    recognizer = Recognizer.load(args.recognizer)
    recognizer.model.summary()
    recognizer.model.layers[0].summary()
    recognizer.model.layers[0].layers[2].summary()


def main():
    args = parse_arguments()

    if args.verbose:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',)

    if args.gpu:
        logging.info("Running in GPU enabled mode.")
        logging.info(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")
        # noinspection PyUnusedLocal
        gpu_owner = safe_gpu.GPUOwner(placeholder_fn=safe_gpu.tensorflow_placeholder)

    if args.mode == "fit":
        fit(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "info":
        info(args)


if __name__ == "__main__":
    main()
