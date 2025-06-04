import itertools
from models.network_builder import build_ann
from models.train import Trainer

class Optimizer:
    def __init__(self, input_shape, output_classes, search_space):
        """
        input_shape: tuple
        output_classes: int
        search_space: dict with keys
            'hidden_layers', 'neurons_per_layer', 'activation', 'learning_rate'
        """
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.search_space = search_space

    def grid_search(self, x_train, y_train, x_val, y_val):
        best_model = None
        best_score = 0.0
        best_params = {}

        combos = list(itertools.product(
            self.search_space['hidden_layers'],
            self.search_space['neurons_per_layer'],
            self.search_space['activation'],
            self.search_space['learning_rate']
        ))

        for hl, neurons, act, lr in combos:
            print(f"Testing: layers={hl}, neurons={neurons}, act={act}, lr={lr}")
            neurons_list = [neurons] * hl
            model = build_ann(
                self.input_shape,
                hidden_layers=hl,
                neurons_per_layer=neurons_list,
                activation=act,
                output_classes=self.output_classes
            )
            trainer = Trainer(model, learning_rate=lr)
            history = trainer.train(x_train, y_train, x_val, y_val)
            val_acc = max(history.history['val_accuracy'])

            if val_acc > best_score:
                best_score = val_acc
                best_model = model
                best_params = {
                    'hidden_layers': hl,
                    'neurons_per_layer': neurons,
                    'activation': act,
                    'learning_rate': lr
                }

        print(f"Best: {best_params} → val_accuracy={best_score:.2%}")
        return best_model, best_params
