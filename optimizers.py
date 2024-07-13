class Optimizer():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def forward():
        pass

class Momentum(Optimizer):
    def __init__(self, lerning_rate = 0.001, gamma = 0.9):
        super().__init__(lerning_rate)   
        self.gamma = gamma
        self.v = 0

    def forward(self, grad_fn, w):
        self.v = self.gamma * self.v + (1 - self.gamma) * self.learning_rate * grad_fn(w)
        return self.v


class NAG(Optimizer):
    def __init__(self, lerning_rate = 0.001, gamma = 0.9):
        super().__init__(lerning_rate)   
        self.gamma = gamma
        self.v = 0

    def forward(self, grad_fn, w):
        self.v = self.gamma * self.v + (1 - self.gamma) * self.learning_rate * grad_fn(w - self.gamma * self.v)
        return self.v
    
