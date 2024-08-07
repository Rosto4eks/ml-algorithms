import numpy as np
from model import BGD

X = np.array([
     [1, 2],
     [0, 0],
     [-4, -8],
     [3, 8],
     [-5, -10],
     [16, 9],
     [5, 5],
     [-2 ,1],
     [-3, -3]
])

Y = np.array([5, 0, -20, 19, -25, 34, 15, 0, -9])

model = BGD(batch_size=1)
model.fit(X, Y)
model.train(iterations=100)
print(model.forward(np.array([100, -100])))