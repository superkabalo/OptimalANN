# config.py

# Dataset settings
DATASET_NAME = 'mnist'          # Options: 'mnist', 'cifar10'

# Network architecture hyperparameters (for grid search)
HIDDEN_LAYERS_OPTIONS = [1, 2, 3]
NEURONS_PER_LAYER_OPTIONS = [32, 64, 128]
ACTIVATION_OPTIONS = ['relu', 'tanh']

# Training hyperparameters
LEARNING_RATE_OPTIONS = [0.01, 0.001]
BATCH_SIZE = 32
EPOCHS = 10

# (Future extensions:)
# OPTIMIZER_OPTIONS = ['adam', 'sgd']
# REGULARIZATION_OPTIONS = [None, 'l2']
