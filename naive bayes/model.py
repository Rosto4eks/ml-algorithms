import numpy as np
from matplotlib import pyplot as plt

# Gaussian Naive Bayes
class GaussianNB:
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.num_classes = len(set(y))
        self.size = X.shape[0]
        self.shape = X.shape[1]

        self.means = []
        self.stds = []
        
        for i in range(self.num_classes):
            arr = self.X[self.y[:] == i]
            self.means.append(arr.mean(axis=0))
            self.stds.append(arr.std(axis=0))
            
    
    def predict(self, x):
        predicts = []
        for i in range(self.num_classes):
            mean = self.means[i]
            std = self.stds[i]

            # pxy = 1
            # for index, feature in enumerate(x):
            #     pxy *= np.exp(- np.square(feature - mean[index]) / (2 * np.square(std[index]))) / np.sqrt(2 * np.pi * np.square(std[index]))
            
            # added log for optimization, log(П [P] ) = E [ log(P) ]
            pxy = 0
            for index, feature in enumerate(x):
                pxy += np.log( np.exp(- np.square(feature - mean[index]) / (2 * np.square(std[index]))) / np.sqrt(2 * np.pi * np.square(std[index])) )

            py = np.sum(self.y == i) / self.size

            predicts.append((pxy * py))

        return np.argmax(predicts), predicts
