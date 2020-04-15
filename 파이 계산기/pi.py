import numpy as np

total = int(1e7)
points = np.random.rand(total, 2)
pi = 4 * np.sum(np.sum(points ** 2, axis=1) < 1) / total
print(pi)
