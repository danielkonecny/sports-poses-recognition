"""
Source: https://www.tensorflow.org/tutorials/images/cnn
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


class ModelCIFAR:
    def __init__(self):
        self.train_images = self.test_images = self.train_labels = self.test_labels = None
        self.model = None

    def load_dataset(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
        # Normalize pixel values to be between 0 and 1
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0

    def verify_data(self):
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i])
            # The CIFAR labels happen to be arrays,
            # which is why you need the extra index
            plt.xlabel(class_names[self.train_labels[i][0]])
        plt.show()

    def create_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.summary()

    def train(self):
        history = self.model.fit(self.train_images,
                                 self.train_labels,
                                 epochs=10,
                                 validation_data=(self.test_images, self.test_labels))

        return history

    def evaluate(self, history):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        test_loss, test_acc = self.model.plot_learning(self.test_images, self.test_labels, verbose=2)

        return test_loss, test_acc


def main():
    model = ModelCIFAR()
    model.load_dataset()
    model.verify_data()
    model.create_model()
    history = model.train()
    loss, acc = model.evaluate(history)

    print(acc)


if __name__ == "__main__":
    main()
