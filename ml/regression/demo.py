import math

import torch
import numpy as np
import matplotlib.pyplot as plt


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x-mu)**2)

x = np.arange(-7, 7, 0.01)

params = [(0, 1), (0, 2), (3, 1)]

plt.figure(figsize=(10, 6))
plt.title('Normal Distribution Curves')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid()
for mu, sigma in params:
    y = []
    for i in x:
        y.append(normal(i, mu, sigma))
    plt.plot(x, y, label=[f'mean {mu}, std {sigma}'])
plt.legend()
plt.show()
