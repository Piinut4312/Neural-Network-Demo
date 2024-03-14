import numpy as np
from learning_rate_scheduler import LRScheduler
from function import Function, Tanh, EPS

"""
This is the old Perceptron code.
Since I've implemented MultilayerPerceptron, this part of code is commented out.

class Perceptron:

    def __init__(self, dim=2):
        self.dim = dim
        self.W = np.random.rand(dim)
        self.b = np.random.rand(1)

    def predict(self, x):
        u = np.dot(self.W, x)+self.b
        return hardlim(u)
    
    def train_single(self, x, y, learning_rate):
        # Update parameters to learn a single point
        # Returns True if the parameters are unchanged, otherwise False is returned instead
        y_pred = self.predict(x)
        error = y-y_pred
        self.W += learning_rate*error*x
        self.b += learning_rate*error
        return error == 0
    
    def train(self, x_list, y_list, epoch=100, learning_rate_scheduler=LRScheduler(1)):
        correct = 0
        for _ in range(epoch):
            correct = 0
            for i in range(len(x_list)):
                if self.train_single(x_list[i], y_list[i], learning_rate_scheduler.step()):
                    correct += 1
            if correct == len(x_list):
                break

    def evaluate(self, x_list, y_list):
        # Return the number of correctly classified data points
        correct = 0
        for i in range(len(x_list)):
            if self.predict(x_list[i]) == y_list[i]:
                correct += 1
        return correct
""" 

class MultiLayerPerceptron:

    def __init__(self, dims:list, activate_func:Function):
        self.dims = dims
        self.W = [np.random.rand(dims[i+1], dims[i]+1)-0.5 for i in range(len(dims)-1)]
        self.v = []     # Save the internal values of each neuron
        self.y = []     # Save the output values of each neuron
        self.delta = []
        self.deltaW = [np.zeros(shape=(dims[i+1], dims[i]+1)) for i in range(len(dims)-1)]
        self.activate_func = activate_func


    def forward(self, input:np.ndarray):
        # Forwarding a given input layer by layer
        self.v = []
        self.y = []
        x = np.array(input)
        for w in self.W:
            x = np.insert(x, 0, -1, axis=0)  # Inserting an additional constant term for bias
            self.y.append(np.array(x))
            x = w @ x
            self.v.append(np.array(x))
            x = self.activate_func(x)

        self.y.append(np.array(x))
        return x
    
    
    def predict(self, input:np.ndarray, classes=2):
        # Predict the label of a given input
        y = self.forward(input)
        y = np.clip(y, EPS, 1-EPS)
        return np.floor(y*classes)/(classes-1)
    
    
    def train_single(self, x_train:np.ndarray, y_train:np.ndarray, learning_rate:float, momentum:float=0.5):
        # Adjusting network parameters using backpropagation algorithm
        # Here, MSE (Mean Square Error) is used as the error function

        y_pred = self.forward(x_train)
        self.delta = [(y_train-y_pred) * self.activate_func.grad(self.v[-1])]
        self.deltaW[-1] = momentum*self.deltaW[-1] + learning_rate*(self.delta[0] @ self.y[-2].transpose())
        self.W[-1] += self.deltaW[-1]
        for i in range(len(self.dims)-3, -1, -1):
            d = (self.delta[-1] @ self.W[i+1][:,1:]) * self.activate_func.grad(self.v[i]).transpose()
            self.delta.append(d)
            self.deltaW[i] = momentum*self.deltaW[i] + learning_rate*(np.outer(d, self.y[i]))
            
            self.W[i] += self.deltaW[i]


    def evaluate(self, x_list, y_list, classes:int=2, metrics=[]):
        # Return the number of correctly classified data points
        y_pred = self.predict(x_list, classes)

        eval_dict = {}
        for metric in metrics:
            eval_dict[metric.name] = metric.compute(y_list, y_pred)

        return eval_dict
    
    
    def compute_mse(self, x_list, y_list):
        return np.mean(np.power(self.forward(x_list)-y_list, 2))
    

    def train(self, x_list, y_list, epoch:int=100, learning_rate_scheduler:LRScheduler=LRScheduler(0.1), momentum:float=0.0, classes:int=2, eval_metrics=[], early_stop_acc=None):
        # Fitting each data one by one using backpropagation algorithm
        self.deltaW = [np.zeros(shape=(self.dims[i+1], self.dims[i]+1)) for i in range(len(self.dims)-1)]
        for i in range(epoch):
            lr = learning_rate_scheduler.step()
            for j in range(x_list.shape[-1]):
                x_train = np.expand_dims(x_list[:, j], axis=1)
                y_train = np.expand_dims(y_list[:, j], axis=1)
                self.train_single(x_train, y_train, lr, momentum)
            
            eval_results = self.evaluate(x_list, y_list, classes, eval_metrics)

            yield i, eval_results

            if "accuracy" in eval_results.keys() and eval_results['accuracy'] >= early_stop_acc:
                break
        