import numpy as np

# Gaussian discriminant analysis
class GDA:
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.num_classes = len(set(y))
        self.size = X.shape[0]
        self.shape = X.shape[1]

        self.means = []
        self.cov_matrices = []
        
        for i in range(self.num_classes):
            arr = self.X[self.y[:] == i]
            self.means.append(arr.sum(axis=0) / len(arr))
            self.cov_matrices.append(np.cov(arr.T))
            
    
    def predict(self, x):
        predicts = []
        for i in range(self.num_classes):
            mean = self.means[i]
            sigma = self.cov_matrices[i]
            
            det = np.linalg.det(sigma)
            inv = np.linalg.inv(sigma)
            norm_coeff = 1.0 / np.sqrt(np.power((2*np.pi), self.shape) * det)
    
            x_mu = np.matrix(x - mean)
            exponent = np.exp( - x_mu * inv * x_mu.T / 2)
            predicts.append((norm_coeff * exponent))

        return np.argmax(predicts), predicts
