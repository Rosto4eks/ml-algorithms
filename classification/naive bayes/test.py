import numpy as np
from matplotlib import pyplot as plt
import NB

x = np.array(
    [[np.random.normal(1.5, 0.5), np.random.normal(1.5, 0.5)] for _ in range(50)] +
    [[np.random.normal(3, 0.4), np.random.normal(3, 0.2)] for _ in range(50)] +
    [[np.random.normal(5, 0.5), np.random.normal(2, 0.5)] for _ in range(60)] +
    [[np.random.normal(3, 0.3), np.random.normal(2, 0.3)] for _ in range(70)] +
    [[np.random.normal(4, 0.5), np.random.normal(0, 0.3)] for _ in range(40)]
)

y = np.array(
    [0 for _ in range(50)] +
    [1 for _ in range(50)] +
    [2 for _ in range(60)] +
    [3 for _ in range(70)] +
    [4 for _ in range(40)]
)

model = NB.GaussianNB()
model.fit(x, y)
print(model.predict([3.1, 3.2]))

plt.plot(x[0:50].T[0], x[0:50].T[1], 'o', color="red")
plt.plot(x[50:100].T[0], x[50:100].T[1], 'o', color="green")
plt.plot(x[100:160].T[0], x[100:160].T[1], 'o', color="blue")
plt.plot(x[160:230].T[0], x[160:230].T[1], 'o', color="yellow")
plt.plot(x[230:270].T[0], x[230:270].T[1], 'o', color="purple")
plt.show()