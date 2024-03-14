import numpy as np
import random

"""
This module contains the DatasetLoader class, which is for reading and parsing raw data from files
"""

class DatasetLoader:

    def __init__(self, file_path, encoding='keep'):
        data_file = open(file_path, mode='r')
        self.data = []
        self.labels = []
        for line in data_file.readlines():
            fields = [float(x) for x in line.split(' ')]
            self.data.append(fields[:-1])
            self.labels.append(fields[-1])
        self.num_cases = len(self.labels)
        self.bound_box = (np.min(self.data, axis=0), np.max(self.data, axis=0))
        self.data = np.array(self.data).transpose()
        self.data_dim = self.data.shape[0]
        self.labels = np.expand_dims(np.array(self.labels), axis=1).transpose()
        if encoding == 'keep':
            pass
        elif encoding == 'scale':
            # Scale the labels into either 0 or 1 with the gap=1/k, where k is the number of classes
            old_labels = list(set(self.labels.flatten()))
            for i in range(len(old_labels)):
                self.labels[self.labels == old_labels[i]] = i/(len(old_labels)-1)
        else:
            raise ValueError("Unknown encoding: "+encoding)
        self.label_set = set(self.labels.flatten())

        
        mean = np.mean(self.data, axis=1, keepdims=True)
        std = np.std(self.data, axis=1, keepdims=True)
        self.standardized_data = (self.data-mean)/std
        self.standardized_bound_box = (np.min(self.standardized_data, axis=1), np.max(self.standardized_data, axis=1))

        data_file.close()

    
    def get_tuples(self, standardize=False):
        if standardize:
            return [(tuple(self.standardized_data[:, i]), float(self.labels[:, i])) for i in range(self.num_cases)]
        else:
            return [(tuple(self.data[:, i]), float(self.labels[:, i])) for i in range(self.num_cases)]


    def sample(self, standardize=False):
        id = np.random.randint(0, self.num_cases)
        if standardize:
            return self.standardized_data[:, id], self.labels[:, id]
        else:
            return self.data[:, id], self.labels[:, id]


    def train_test_split(self, test_size=0.33, standardize=False):
        indices = range(self.num_cases)
        test_indices = random.choices(indices, k=int(self.num_cases*test_size))
        train_indices = [i for i in indices if i not in test_indices]
        random.shuffle(train_indices)
        random.shuffle(test_indices)

        if standardize:
            train_data = self.standardized_data[:, train_indices]
            test_data = self.standardized_data[:, test_indices]
        else:
            train_data = self.data[:, train_indices]
            test_data = self.data[:, test_indices]

        train_labels = self.labels[:, train_indices]
        test_labels = self.labels[:, test_indices]
        return train_data, train_labels, test_data, test_labels