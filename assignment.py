from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
import numpy as np

from beras.model import SequentialModel

def get_model():
    model = SequentialModel(
        [
           # Add in your layers here as elements of the list!
           # e.g. Dense(10, 10),
           Dense(784, 200),
           Softmax(),
           Dense(200, 100),
           Softmax(),
           Dense(100, 10),
           Softmax(),
        ]
    )
    return model

def get_optimizer():
    # choose an optimizer, initialize it and return it!
    return Adam(0.001)

def get_loss_fn():
    # choose a loss function, initialize it and return it!
    loss = CategoricalCrossEntropy()
    return loss

def get_acc_fn():
    # choose an accuracy metric, initialize it and return it!
    return CategoricalAccuracy()

if __name__ == '__main__':

    ### Use this area to test your implementation!

    # 1. Create a SequentialModel using get_model
    model = get_model()
    # 2. Compile the model with optimizer, loss function, and accuracy metric
    model.compile(get_optimizer(),get_loss_fn(),get_acc_fn())
    # 3. Load and preprocess the data
    OneHotEncode = OneHotEncoder()
    train_inputs, train_labels, test_inputs, test_labels = load_and_preprocess_data()
    aaa = np.concatenate([train_labels,test_labels],axis = -1)
    OneHotEncode.fit(data = aaa)
    # 4. Train the model
    model.fit(train_inputs,OneHotEncode.forward(train_labels),10,5)
    # 5. Evaluate the model
    model.evaluate(test_inputs, OneHotEncode.forward(test_labels), 5)

    
