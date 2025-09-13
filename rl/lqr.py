import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
import math

class VehicleModel:
    def __init__(self, x, y, psi, v, L, dt):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.L = L
        self.dt = dt

    def update_state(self, a, delta_f):
        self.x += self.v * np.cos(self.psi) * self.dt
        self.y += self.v * np.sin(self.psi) * self.dt
        self.psi += self.v /self.L * np.tan(delta_f) * self.dt
        self.v += a * self.dt

    def get_state(self):
        return self.x, self.y, self.psi, self.v

    def state_space(self, ref_delta, ref_yaw):
        # A = np.array([[1.0, 0.0, -self.v * self.dt *])
        pass

class LQR:
    def __init__(self):
        pass

    def calRicatti(self, A, B, Q, R):
        max_iter = 500
        eps = 0.01
        Qf = Q
        P = Qf
        for iter in range(max_iter):
            P = solve_continuous_are(A, B, Q, R)
            if iter > 0 and np.max(np.abs(P - P_prev)) < eps:
                break
            P_prev = P
        return P

    def lqrcontrol(self, robot_state, refer_point, A, B, Q, R):
        x = robot_state[0:3] - refer_point[0:3]
        P = self.calRicatti(A, B, Q, R)
        K = -np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
        u = K @ x
        u_star = u
        return u_star[0,1]


def GenerateReferenceLine():
    # 使用五次多项式生成参考路径
    x = np.linspace(0, 50, 500)
    # 五次多项式系数：y = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f
    # 这里我们假设一些边界条件来求解多项式系数
    # 假设起点 (0, 0)，终点 (50, 10)，初始斜率0，终点斜率0，初始曲率0，终点曲率0
    
    # 构建系数矩阵和结果向量
    A = np.array([
        [0, 0, 0, 0, 0, 1],           # y(0) = 0
        [50**5, 50**4, 50**3, 50**2, 50, 1],  # y(50) = 10
        [0, 0, 0, 0, 1, 0],           # y'(0) = 0
        [5*50**4, 4*50**3, 3*50**2, 2*50, 1, 0], # y'(50) = 0
        [0, 0, 0, 2, 0, 0],           # y''(0) = 0
        [20*50**3, 12*50**2, 6*50, 2, 0, 0]   # y''(50) = 0
    ])
    b = np.array([0, 10, 0, 0, 0, 0])
    
    # 求解系数
    coeffs = np.linalg.solve(A, b)
    a, b, c, d, e, f = coeffs
    
    # 计算y值
    y = a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

    # 计算theta角


    # 计算曲率k

    
    # 返回参考路径点
    ref_line = np.column_stack((x, y))
    return ref_line

def calcNearestPoint(x, y, reference_line):
    min_d = float('inf')
    nearest_point = [x, y]
    for [rx, ry] in  reference_line:
        dx = x - rx
        dy = y - ry
        d = np.hypot(dx, dy)
        if d < min_d:
            min_d = d
            nearest_point = [rx, ry]
    return nearest_point, min_d

if __name__ == "__main__":
    ugv = VehicleModel(0, 0, 0, 0, 1, 0.1)
    lqr = LQR()
    ref_line = GenerateReferenceLine()
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.plot(ref_line[:, 0], ref_line[:, 1], color='blue')

    init_state = [1, 0]
    for i in range(500):
        robot_state = np.zeros(4)
        robot_state[0] = ugv.x
        robot_state[1] = ugv.y
        robot_state[2] = ugv.psi
        robot_state[3] = ugv.v
        e, k, ref_yaw= calcNearestPoint(robot_state[0], robot_state[1], ref_line)
        ref_delta = math.atan2(L*k, 1)
        A, B = ugv.state_space(ref_delta, ref_yaw)
        delta = lqr.lqrcontrol(robot_state, )
        delta += ref_delta
        ugv.update_state(0, delta)

    plt.show()


