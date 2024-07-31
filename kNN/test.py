import numpy as np
from .model import kNN, AnnoyForest

X = np.array(
    [[np.random.normal(1.5, 0.6), np.random.normal(1.5, 0.6)] for _ in range(50)] +
    [[np.random.normal(2, 0.6), np.random.normal(2, 2.2)] for _ in range(50)]
)

y = np.array(
    [0 for _ in range(50)] +
    [1 for _ in range(50)]
)

model = kNN(10)
model.fit(X, y)
print(model.predict(np.array([2.5, 2.0])))

# a = AnnoyForest()
# a.fit(X, y)
# print(a.predict(np.array([1.5, 1.5])))