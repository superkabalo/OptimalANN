import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
import config

class Trainer:
    def __init__(self, model, learning_rate=0.001):
        """
        model: keras model
        learning_rate: float
        """
        self.model = model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, x_train, y_train, x_val, y_val):
        """
        Train using config.BATCH_SIZE and config.EPOCHS
        """
        return self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            verbose=1
        )

    def evaluate(self, x_test, y_test):
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Accuracy: {acc*100:.2f}%")
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def get_accuracy(self, y_true, y_pred):
        y_pred_labels = tf.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred_labels)
