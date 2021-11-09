"""
Source: https://www.tensorflow.org/tutorials/images/transfer_learning
"""

import matplotlib.pyplot as plt
import os
import tensorflow as tf


def plot_learning(history, acc=None, val_acc=None, loss=None, val_loss=None, initial_epochs=0):
    if acc is None:
        acc = []
    if val_acc is None:
        val_acc = []
    if loss is None:
        loss = []
    if val_loss is None:
        val_loss = []

    acc += history.history['accuracy']
    val_acc += history.history['val_accuracy']

    loss += history.history['loss']
    val_loss += history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.ylabel('Accuracy')
    if initial_epochs != 0:
        plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.ylabel('Cross Entropy')
    if initial_epochs != 0:
        plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    return acc, val_acc, loss, val_loss


class ModelTL:
    def __init__(self):
        self.img_size = (160, 160)
        self.initial_epochs = 10

        self.train_dataset = self.validation_dataset = self.test_dataset = None
        self.class_names = None
        self.base_model = self.model = None

    def load_dataset(self):
        _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
        path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
        path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

        train_dir = os.path.join(path, 'train')
        validation_dir = os.path.join(path, 'validation')

        batch_size = 32

        self.train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                         shuffle=True,
                                                                         batch_size=batch_size,
                                                                         image_size=self.img_size)

        self.validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                              shuffle=True,
                                                                              batch_size=batch_size,
                                                                              image_size=self.img_size)

        val_batches = tf.data.experimental.cardinality(self.validation_dataset)
        self.test_dataset = self.validation_dataset.take(val_batches // 5)
        self.validation_dataset = self.validation_dataset.skip(val_batches // 5)

        self.class_names = self.train_dataset.class_names

        print(f'Number of validation batches: {tf.data.experimental.cardinality(self.validation_dataset)}')
        print(f'Number of test batches: {tf.data.experimental.cardinality(self.test_dataset)}')

        auto_tune = tf.data.AUTOTUNE

        self.train_dataset = self.train_dataset.prefetch(buffer_size=auto_tune)
        self.validation_dataset = self.validation_dataset.prefetch(buffer_size=auto_tune)
        self.test_dataset = self.test_dataset.prefetch(buffer_size=auto_tune)

    def display_dataset(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_dataset.take(1):
            for i in range(9):
                _ = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")

    def create_model(self):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
        ])
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        # Create the base model from the pre-trained model MobileNet V2
        img_shape = self.img_size + (3,)
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                            include_top=False,
                                                            weights='imagenet')

        image_batch, label_batch = next(iter(self.train_dataset))
        feature_batch = self.base_model(image_batch)
        print(feature_batch.shape)

        self.base_model.trainable = False
        self.base_model.summary()

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(1)

        inputs = tf.keras.Input(shape=(160, 160, 3))
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = self.base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        self.model = tf.keras.Model(inputs, outputs)

        base_learning_rate = 0.0001
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.summary()

    def train(self):
        loss0, accuracy0 = self.model.plot_learning(self.validation_dataset)
        print(f"initial loss: {loss0:.2f}")
        print(f"initial accuracy: {accuracy0:.2f}")

        history = self.model.fit(self.train_dataset,
                                 epochs=self.initial_epochs,
                                 validation_data=self.validation_dataset)

        return history

    def prepare_fine_tuning(self):
        self.base_model.trainable = True

        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(self.base_model.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = 100

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        base_learning_rate = 0.0001
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / 10),
                           metrics=['accuracy'])

        self.model.summary()

    def fine_tune(self, history):
        fine_tune_epochs = 10
        total_epochs = self.initial_epochs + fine_tune_epochs

        history_fine = self.model.fit(self.train_dataset,
                                      epochs=total_epochs,
                                      initial_epoch=history.epoch[-1],
                                      validation_data=self.validation_dataset)

        return history_fine

    def evaluate(self):
        loss, accuracy = self.model.plot_learning(self.test_dataset)
        print('Test accuracy :', accuracy)

        # Retrieve a batch of images from the test set
        image_batch, label_batch = self.test_dataset.as_numpy_iterator().next()
        predictions = self.model.predict_on_batch(image_batch).flatten()

        # Apply a sigmoid since our model returns logits
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)

        print('Predictions:\n', predictions.numpy())
        print('Labels:\n', label_batch)

        plt.figure(figsize=(10, 10))
        for i in range(9):
            _ = plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].astype("uint8"))
            plt.title(self.class_names[predictions[i]])
            plt.axis("off")


def main():
    model = ModelTL()
    model.load_dataset()
    model.display_dataset()
    model.create_model()
    history = model.train()
    acc, val_acc, loss, val_loss = plot_learning(history)

    model.prepare_fine_tuning()
    history_fine = model.fine_tune(history)
    plot_learning(history_fine, acc, val_acc, loss, val_loss, model.initial_epochs)


if __name__ == "__main__":
    main()
