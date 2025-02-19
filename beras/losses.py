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
        return np.mean(np.square(np.clip(y_pred,1e-8, 1-1e-8) - y_true))

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs[0], self.inputs[1]  
        batch_size = y_pred.shape[0]
        grad = (2 / batch_size) * (np.clip(y_pred,1e-8, 1-1e-8) - y_true)
        return [grad]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        loss = y_true * np.log(np.clip(y_pred,1e-8, 1-1e-8))

        return -np.mean(np.sum(loss,-1))

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs[0], self.inputs[1]  
        batch_size = y_pred.shape[0]
        grad = -y_true / (np.clip(y_pred,1e-8, 1-1e-8) * batch_size)

        return [grad]
