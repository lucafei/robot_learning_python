import numpy as np
import matplotlib.pyplot as plt

def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

def binomialCoefficient(n, i):
    if n==i or i==0:
        return 1
    return factorial(n) / (factorial(i) * factorial(n-i))

def bernsteinBasis(n, i, t):
    return binomialCoefficient(n, i) * pow(t, i) * pow(1-t, n-i)

class Bezier:
    def __init__(self, order, control_points):
        self.m_order = order
        self.m_control_points = control_points

    def at(self, t):
        x = 0.0
        y = 0.0
        for i in range(self.m_order + 1):
            basis = bernsteinBasis(self.m_order, i, t)
            x += basis * self.m_control_points[i][0]
            y += basis * self.m_control_points[i][1]
        return [x, y]

if __name__ == "__main__":
    a = [0,0.3]
    c = [0.5,0.5]
    k = [1.5,0.55]
    b = [4,0.5]
    d = [7,0.3]
    points = []
    points.append(a)
    points.append(c)
    points.append(k)
    points.append(b)
    points.append(d)
    bezier = Bezier(4, points)
    x = []
    y = []
    for t in np.linspace(0, 1, 100):
        point = bezier.at(t)
        x.append(point[0])
        y.append(point[1])

    plt.plot(x, y)
    plt.scatter(*zip(*points), color='red')  # 可选：显示控制点
    plt.show()