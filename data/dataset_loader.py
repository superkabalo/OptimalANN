import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
import config

class DatasetLoader:
    def __init__(self, dataset_name=None):
        """
        Initialize the dataset loader.

        Parameters:
        - dataset_name (str): Name of the dataset to load. Options: 'mnist', 'cifar10'.
                              If None, uses config.DATASET_NAME.
        """
        self.dataset_name = (dataset_name or config.DATASET_NAME).lower()

    def load_data(self):
        """
        Load the selected dataset and preprocess it.

        Returns:
        - (x_train, y_train), (x_val, y_val), (x_test, y_test)
        """
        if self.dataset_name == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            input_shape = (28, 28)
        elif self.dataset_name == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            input_shape = (32, 32, 3)
        else:
            raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar10'.")

        # Normalize to [0,1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Reshape for the model
        x_train = x_train.reshape((-1, ) + input_shape)
        x_test = x_test.reshape((-1, ) + input_shape)

        # One-hot encode labels
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Split off 20% of training as validation
        split = int(0.8 * len(x_train))
        x_val, y_val = x_train[split:], y_train[split:]
        x_train, y_train = x_train[:split], y_train[:split]

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
