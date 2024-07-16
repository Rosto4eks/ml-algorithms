import numpy as np
from model import ID3, tree_runner

x = np.array([
    ["sunny", "hot", "high", "no"],
    ["sunny", "hot", "high", "yes"],
    ["overcast", "hot", "high", "no"],
    ["rain", "mild", "high", "no"],
    ["rain", "cool", "medium", "no"],
    ["rain", "cool", "medium", "yes"],
    ["overcast", "cool", "medium", "yes"],
    ["sunny", "mild", "high", "no"],
    ["sunny", "cool", "medium", "no"],
    ["rain", "mild", "medium", "no"],
    ["sunny", "mild", "medium", "yes"],
    ["overcast", "mild", "high", "yes"],
    ["overcast", "hot", "medium", "no"],
    ["rain", "mild", "high", "yes"],
])

y = np.array([
    "no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"
])

model = ID3()
model.fit(x, y)
model.train()

print(model.predict(["sunny", "hot", "medium", "yes"]))
