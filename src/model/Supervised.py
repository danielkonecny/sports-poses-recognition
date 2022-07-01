"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for recognizing sports poses trained with supervision.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 01. 07. 2022
"""

from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf
import pandas as pd

from safe_gpu import safe_gpu


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'dataset',
        type=str,
        help="Location of the directory with dataset.",
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
        default=.2,
        help="Number between 0 and 1 representing proportion of dataset to be used for validation."
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=5,
        help="Number of epochs to be performed on a dataset for fitting."
    )
    parser.add_argument(
        '-p', '--dataset_portion',
        type=float,
        default=1.,
        help="Portion of dataset that is used, number between 0 and 1 (0 not included)."
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
        help="Use to turn on exporting of validation accuracy to file logs/accuracies.csv."
    )
    parser.add_argument(
        '-t', '--top_k_accuracy',
        nargs="+",
        type=int,
        default=[1],
        help="Evaluate top K accuracy of the model, allows for multiple Ks."
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


def export_accuracies(fields):
    log_file_path = Path("logs/accuracies.csv")
    field_names = list(fields.keys())

    if log_file_path.exists():
        accuracy_df = pd.read_csv(log_file_path)
        for field_name in field_names:
            if field_name not in accuracy_df.columns:
                print("Su - Accuracy logging file header is not corresponding. "
                      "Saving the file as .bak and creating a new one.")
                accuracy_df.to_csv(log_file_path.parent / (log_file_path.stem + '.bak.csv'), index=False)
                accuracy_df = pd.DataFrame(columns=field_names)
    else:
        accuracy_df = pd.DataFrame(columns=field_names)

    new_row = pd.DataFrame(fields, index=[0])
    accuracy_df = pd.concat([accuracy_df, new_row])
    accuracy_df.to_csv(log_file_path, index=False)


def info():
    model = tf.keras.applications.resnet50.ResNet50(weights=None, classes=4)
    model.summary()


def main():
    args = parse_arguments()

    if args.gpu:
        # noinspection PyUnusedLocal
        gpu_owner = safe_gpu.GPUOwner(placeholder_fn=safe_gpu.tensorflow_placeholder)

    top_k_accuracies = []
    for top_k in args.top_k_accuracy:
        top_k_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=top_k, name=f'top_{top_k}_categorical_accuracy')
        top_k_accuracies.append(top_k_accuracy)

    dataset = Path(args.dataset)
    labels = [x.stem for x in sorted(dataset.iterdir()) if x.is_dir()]
    model = tf.keras.applications.resnet50.ResNet50(
        weights=None,
        classes=len(labels)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=top_k_accuracies
    )

    if args.seed is None:
        random_seed = tf.random.uniform(shape=(), minval=1, maxval=2 ** 32, dtype=tf.int64)
    else:
        random_seed = args.seed
    # TODO - TF 2.10 enables to return train and val at once as a tuple, shorten then.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset,
        label_mode='categorical',
        batch_size=args.batch_size,
        image_size=(args.height, args.width),
        seed=random_seed,
        validation_split=args.validation_split,
        subset="training"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset,
        label_mode='categorical',
        batch_size=args.batch_size,
        image_size=(args.height, args.width),
        seed=random_seed,
        validation_split=args.validation_split,
        subset="validation"
    )
    # FIXME - Sharding causes validation TF.data.Dataset to somehow not be consistent.
    if args.dataset_portion < 1.:
        train_ds = train_ds.shard(num_shards=tf.cast(1 / args.dataset_portion, tf.int64), index=0)
        val_ds = val_ds.shard(num_shards=tf.cast(1 / args.dataset_portion, tf.int64), index=0)

    print(f'Su - Number of train batches (size {args.batch_size}) loaded: '
          f'{tf.data.experimental.cardinality(train_ds)}.')
    print(f'Su - Number of validation batches (size {args.batch_size}) loaded: '
          f'{tf.data.experimental.cardinality(val_ds)}.')

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds
    )

    epoch = tf.math.argmax(history.history[f'val_top_{args.top_k_accuracy[0]}_categorical_accuracy']) + 1
    val_accuracy = tf.math.reduce_max(history.history[f'val_top_{args.top_k_accuracy[0]}_categorical_accuracy'])
    print(f"Su - Best validation accuracy {val_accuracy:.4f} in epoch {epoch:02d}.")

    if args.export_accuracy:
        fields = {
            'Model': 'Supervised',
            'Training Data Portion': f'{1 - args.validation_split}',
            'Seed': f'{args.seed}'
        }
        for k in args.top_k_accuracy:
            fields[f'Top-{k} Validation Accuracy'] = \
                f'{history.history[f"val_top_{k}_categorical_accuracy"][epoch - 1]:.4f}'
        export_accuracies(fields)


if __name__ == "__main__":
    main()
    # info()
