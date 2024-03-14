import numpy as np

# This module defines the evaluation metrics

class Metric:

    def __init__(self, name, func):
        self.name = name
        self.func = func

    def compute(self, y_true, y_pred):
        return self.func(y_true, y_pred)

def accuracy(y_true, y_pred):
    # Return the accuracy of the prediction
    # Accuracy is defined as (Number of correct prediction)/(Total number of predictions)
    if y_true.shape != y_pred.shape:
        raise ValueError("The shape of y_true and y_pred should be the same.")
    num_correct = np.sum((y_pred == y_true).astype(np.int8))
    return num_correct / y_true.shape[-1]


ACCURACY = Metric("accuracy", accuracy)