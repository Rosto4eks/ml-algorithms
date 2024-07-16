import numpy as np
from matplotlib import pyplot as plt
from model import LogRegression

x = np.array(
    [[np.random.normal(1.5, 0.5), np.random.normal(1.5, 0.5)] for _ in range(50)] +
    [[np.random.normal(3, 0.4), np.random.normal(3, 0.2)] for _ in range(50)]
)

y = np.array(
    [0 for _ in range(50)] +
    [1 for _ in range(50)]
)

model = LogRegression()
model.fit(x, y)
model.train()

print(model.w)
x1 = 0
x2 = 4
y1 = (-model.w[1] * x1 - model.w[0]) / model.w[2]
y2 = (-model.w[1] * x2 - model.w[0]) / model.w[2]

plt.plot([x1, x2], [y1, y2])
plt.plot(x[0:50].T[0], x[0:50].T[1], 'o', color="red")
plt.plot(x[50:100].T[0], x[50:100].T[1], 'o', color="blue")
plt.show()