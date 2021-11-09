"""
Source: https://www.tensorflow.org/tutorials/images/classification
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def download_dataset():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f"Image Count: {image_count}")

    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0]))
    PIL.Image.open(str(roses[1]))

    tulips = list(data_dir.glob('tulips/*'))
    PIL.Image.open(str(tulips[0]))
    PIL.Image.open(str(tulips[1]))

    return data_dir


class ModelFlower:
    def __init__(self):
        self.img_height = 180
        self.img_width = 180
        self.epochs = 15

        self.train_ds = self.val_ds = None
        self.model = None
        self.class_names = None

    def create_dataset(self, data_dir):
        batch_size = 32

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size)

        self.class_names = self.train_ds.class_names
        print(f"Class Names: {self.class_names}")

        for image_batch, labels_batch in self.train_ds:
            print(f"Image Batch Shape: {image_batch.shape}")
            print(f"Labels Batch Shape: {labels_batch.shape}")
            break

    def visualize_dataset(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(9):
                _ = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")

    def configure_dataset(self):
        auto_tune = tf.data.AUTOTUNE

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=auto_tune)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=auto_tune)

    def create_model(self):
        num_classes = 5

        data_augmentation = Sequential([
            layers.RandomFlip("horizontal", input_shape=(self.img_height, self.img_width, 3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        self.model = Sequential([
            data_augmentation,
            layers.Rescaling(1. / 255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.summary()

    def train(self):
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

        return history

    def evaluate(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def predict(self):
        sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
        sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

        img = tf.keras.utils.load_img(
            sunflower_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(f"This image most likely belongs to {self.class_names[np.argmax(score)]}"
              f" with a {100 * np.max(score):.2f} percent confidence.")


def main():
    data_dir = download_dataset()

    model = ModelFlower()
    model.create_dataset(data_dir)
    model.visualize_dataset()
    model.configure_dataset()
    model.create_model()
    history = model.train()
    model.evaluate(history)
    model.predict()


if __name__ == "__main__":
    main()
