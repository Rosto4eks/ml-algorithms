import numpy as np

class LinearSVM:
    def fit(self, x, y, C = 1.0):
        self.x = x
        self.y = y
        self.C = C

        self.size = x.shape[0]
        self.n_features = x.shape[1]

        self.w = np.random.randn(self.n_features)
        self.b = 0

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b)

    def margin(self, x, y):
        return y * (self.w.dot(x) + self.b)

    def loss(self):
        hinge_loss = np.maximum(0, 1 - self.margin(self.x, self.y)).mean()
        return hinge_loss + 0.5 * np.dot(self.w, self.w)
    
    def diff(self, x, y):
        return - self.C * y * x
    
    def diff_b(self, x, y):
        return - self.C * y

    def train(self, epoch=1000, learning_rate=0.01, batch_size=20):
        for e in range(epoch):
            count = 0
            w = 0
            b = 0
            for x,y in zip(self.x, self.y):
                if 1 - self.margin(x, y) > 0:
                    w += learning_rate * self.diff(x, y)
                    b += learning_rate * self.diff_b(x, y)

                if count >= batch_size:
                    self.w -= w
                    self.b -= b
                    count = -1
                    w = 0
                    b = 0
                count += 1
