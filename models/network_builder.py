import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_ann(input_shape, hidden_layers, neurons_per_layer,
              activation='relu', output_classes=10):
    """
    Builds an Artificial Neural Network (ANN) with customizable parameters.

    Parameters:
    - input_shape (tuple): e.g. (28,28) or (32,32,3)
    - hidden_layers (int)
    - neurons_per_layer (list[int])
    - activation (str)
    - output_classes (int)

    Returns:
    - keras.Sequential model (uncompiled)
    """
    assert len(neurons_per_layer) == hidden_layers, (
        "len(neurons_per_layer) must equal hidden_layers"
    )

    model = keras.models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())

    for n in neurons_per_layer:
        model.add(layers.Dense(n, activation=activation))

    model.add(layers.Dense(output_classes, activation='softmax'))
    return model
