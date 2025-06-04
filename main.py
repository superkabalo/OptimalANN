import numpy as np
import config
from data.dataset_loader import DatasetLoader
from models.optimizer import Optimizer
from models.train import Trainer

def main():
    # 1. Load data per config
    loader = DatasetLoader(config.DATASET_NAME)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = loader.load_data()

    # 2. Build search space from config
    search_space = {
        'hidden_layers': config.HIDDEN_LAYERS_OPTIONS,
        'neurons_per_layer': config.NEURONS_PER_LAYER_OPTIONS,
        'activation': config.ACTIVATION_OPTIONS,
        'learning_rate': config.LEARNING_RATE_OPTIONS
    }

    # 3. Optimize
    input_shape = x_train.shape[1:]
    output_classes = y_train.shape[1]
    optimizer = Optimizer(input_shape, output_classes, search_space)
    best_model, best_params = optimizer.grid_search(
        x_train, y_train, x_val, y_val
    )
    print(f"\nBest Model Parameters: {best_params}")

    # 4. Retrain best on full train+val
    trainer = Trainer(best_model, learning_rate=best_params['learning_rate'])
    trainer.train(np.concatenate([x_train, x_val]),
                  np.concatenate([y_train, y_val]),
                  x_val, y_val)  # optional

    # 5. Final evaluation
    test_acc = trainer.evaluate(x_test, y_test)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
