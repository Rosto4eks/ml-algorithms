import numpy as np

class LinearSVM:
    def fit(self, X, y, C = 1.0):
        self.X = X
        self.y = y
        self.C = C

        self.size = X.shape[0]
        self.n_features = X.shape[1]

        self.w = np.random.randn(self.n_features)
        self.b = 0

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b)

    def margin(self, x, y):
        return y * (self.w.dot(x) + self.b)

    def loss(self):
        return np.maximum(0, 1 - self.margin(self.X, self.y)).mean()  + 0.5 * np.dot(self.w, self.w)
    
    def diff(self, x, y):
        return - self.C * y * x


    def train(self, epoch=1000, learning_rate=0.01, batch_size=20):
        for e in range(epoch):
            count = 0
            w = 0
            b = 0
            for x,y in zip(self.X, self.y):
                if 1 - self.margin(x, y) > 0:
                    w += learning_rate * self.diff(x, y)
                    b += learning_rate * self.diff(1, y)

                if count >= batch_size:
                    self.w -= w
                    self.b -= b
                    count = -1
                    w = 0
                    b = 0
                count += 1

class SVM:
    def __init__(self, kernel = "linear", \
                 C = 1, max_passes = 100, \
                 tol = 1e-3, poly_degree = 3, \
                 rbf_sigma = 1 / np.exp(1) / 4, sigm_a = 0.01, sigm_b = 2):
        self.kernel = kernel
        self.C = C
        self.max_passes = max_passes
        self.tol = tol
        self.poly_degree = poly_degree
        self.rbf_sigma = rbf_sigma
        self.sigm_a = sigm_a
        self.sigm_b = sigm_b


    def fit(self, X, y):
        self.X = X
        self.y = y
        self.y[self.y == 0] = -1
        self.b = 0

        self.size = X.shape[0]
        self.n_features = X.shape[1]

        self.alphas = np.zeros(self.size)
        self.K = self.kernel_function(X, X)
        
        passes = 0
        while passes < self.max_passes:
            changed_alphas = 0

            for i in range(self.size):
                x_i = self.X[i]
                y_i = self.y[i]
                a_i = self.alphas[i]

                err_i = self.decision_function(x_i) - y_i

                if (y_i * err_i < - self.tol and a_i < self.C) or (y_i * err_i > self.tol and a_i > 0):
                    j = i
                    while j == i:
                        j = np.random.randint(self.size)
                    x_j = self.X[j]
                    y_j = self.y[j]
                    a_j = self.alphas[j]

                    err_j = self.decision_function(x_j) - y_j
                    a_i_old = a_i
                    a_j_old = a_j

                    l, h = self.boundaries(y_i, y_j, a_i, a_j)
                    if l == h:
                        continue

                    eta =  2 * self.kernel_function(x_i, x_j) - self.kernel_function(x_i, x_i) - self.kernel_function(x_j, x_j)
                    if eta >= 0:
                        continue

                    a_j = a_j - y_j * (err_i - err_j) / eta
                    a_j = np.clip(a_j, l, h)

                    if (np.abs(a_j - a_j_old) < 1e-5):
                        continue

                    a_i = a_i + y_i * y_j * (a_j_old - a_j)

                    self.alphas[i] = a_i
                    self.alphas[j] = a_j

                    b_1 = self.b - err_i - y_i * (a_i - a_i_old) * self.K[i, i] - y_j * (a_j - a_j_old) * self.K[i, j]
                    b_2 = self.b - err_j - y_i * (a_i - a_i_old) * self.K[i, j] - y_j * (a_j - a_j_old) * self.K[j, j]

                    if 0 < a_i < self.C:
                        self.b = b_1
                    elif 0 < a_j < self.C:
                        self.b = b_2
                    else:
                        self.b = (b_1 + b_2) / 2

                    changed_alphas += 1
            
            if changed_alphas == 0:
                passes += 1

    def check_kkt(self, x):
        pass

    def boundaries(self, y_1, y_2, a_1, a_2):
        if y_1 == y_2:
            l = np.max([0, a_1 + a_2 - self.C])
            h = np.min([self.C, a_1 + a_2])
        else:
            l = np.max([0, a_2 - a_1])
            h = np.min([self.C, self.C + a_2 - a_1])
        return l, h

    def decision_function(self, x):
        return np.dot(self.alphas * self.y, self.kernel_function(self.X, x)) + self.b
    
    def predict(self, x):
        return np.sign(self.decision_function(x))


    def kernel_function(self, x_1, x_2):
        if self.kernel == "linear":
            return np.dot(x_1, x_2.T)
        if self.kernel == "poly":
            return np.power(np.dot(x_1, x_2.T) + 2, self.poly_degree)
        if self.kernel == "sigmoid":
            return np.tanh(self.sigm_a * np.dot(x_1, x_2.T) + self.sigm_b)
        if self.kernel == "rbf":
            if np.ndim(x_1) == 1:
                x_1 = x_1[np.newaxis, :]

            if np.ndim(x_2) == 1:
                x_2 = x_2[np.newaxis, :]

            dif = np.squeeze(x_1[:, np.newaxis, :] - x_2[np.newaxis, :, :])
            dist_squared = np.linalg.norm(dif, axis = dif.ndim - 1) ** 2
            return np.exp(- self.rbf_sigma * dist_squared)
            


    
