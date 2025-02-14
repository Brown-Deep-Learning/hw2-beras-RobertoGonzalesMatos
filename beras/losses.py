import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return np.mean(np.square(y_pred - y_true))

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs[0], self.inputs[1]  
        batch_size = y_pred.shape[0]  # Proper shape extraction
        grad = (2 / batch_size) * (y_pred - y_true)
        return [grad]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        loss = -np.sum(y_true * np.log(y_pred), axis=-1)

        return np.mean(loss)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs[0], self.inputs[1]  
        batch_size = y_pred.shape[0]
        grad = (y_pred - y_true) / batch_size

        return [grad]
