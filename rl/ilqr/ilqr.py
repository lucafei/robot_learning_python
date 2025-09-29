from vehicle_dynamics import *

from cost_function import CostFunc
from rs import *

'''
https://github.com/TeleYuhao/AL-ILQR/blob/main/ilqr.py
'''

class iLQR:
    def __init__(self, vehicle_dynamics, cost_function, ref_state):
        self.vehicle_dynamic = vehicle_dynamics
        self.cost_func = cost_function
        self.ref_state = ref_state
        self.step = len(ref_state) - 1
        self.max_iter = 50
        self.line_search_beta_1= 1e-4
        self.line_search_beta_2 = 10
        self.line_search_gamma = 0.5
        self.J_tolerance = 1e-2


    def init_trajectory(self, x, u):
        x_init = []
        for i in range(self.step):
            x_init.append(self.vehicle_dynamic.RK3Function(x_init[-1], u[i]))
        return x_init

    def evaluate(self, x, u):
        return self.cost_func.calc_cost(x,u)
    def solve(self, init_state, init_control):
        u = init_control
        x = init_state
        J_opt = self.evaluate(x, u)
        J_hist = [J_opt]
        x_hist = [x]
        u_hist = [u]
        iter = 0
        converged = False
        while not converged:
            print("ILQR: New Iteration {}".format(iter))
            if iter >= 500:
                print("iLQR: Max iteration reached")
                break
            k, d, Qu_list, Quu_list = self.backward(x, u)
            alpha = 1
            J_new = 0
            accept = False
            while not accept:
                x_new, u_new, J_new, delta_J = self.forward(x, u ,k, d, alpha, Qu_list, Quu_list)
                z = (J_opt - J_new) / -delta_J
                if self.line_search_beta_1 < z < self.line_search_beta_2:
                    x = x_new
                    u = u_new
                    accept = True
                alpha *= self.line_search_gamma
            iter += 1
            J_hist.append(J_opt)
            x_hist.append(x)
            u_hist.append(u)
            if accept:
                if abs(J_opt - J_new)/J_opt < self.J_tolerance:
                    converged = True
                    print("iLQR: Converged at iteration {}".format(iter))
                J_opt = J_new
        for i in range(1, len(x_hist[-1])):
            x_hist[-1][i] = x_hist[-1][i].full().T.flatten()

        res_dict = {'x_hist': x_hist, 'u_hist': u_hist, 'J_hist': J_hist}
        return res_dict


    def backward(self, x, u):
        p = self.cost_func.p_fun(self.ref_state[-1], x[-1])
        P = self.cost_func.P_fun(self.ref_state[-1], x[-1])

        k = [None] * self.step
        d = [None] * self.step

        Qu_list = [None] * self.step
        Quu_list = [None] * self.step

        for i in reversed(range(self.step)):
            dfdx = self.vehicle_dynamic.dfdx_func(x[i], u[i])
            dfdu = self.vehicle_dynamic.dfdu_func(x[i], u[i])
            lx = self.cost_func.lx_fun(self.ref_state[i], x[i], u[i])
            lu = self.cost_func.lu_fun(self.ref_state[i], x[i], u[i])
            lxx = self.cost_func.lxx_fun(self.ref_state[i], x[i], u[i])
            lux = self.cost_func.lux_fun(self.ref_state[i], x[i], u[i])
            luu = self.cost_func.luu_fun(self.ref_state[i], x[i], u[i])

            Qx = lx + p @ dfdx
            Qu = lu + p @ dfdu
            Qxx = lxx + dfdx.T @ P @ dfdx
            Qux = lux + dfdu.T @ P @ dfdx
            Quu = luu + dfdu.T @ P @ dfdu

            Quu_inverse = regularized_persudo_inverse(Quu)
            # Quu_inverse = np.linalg.inv(Quu)
            Quu_list[i] = Quu
            Qu_list[i] = Qu

            k[i] = - Quu_inverse @ Qux
            d[i] = - Quu_inverse @ Qu.T

            p = Qx + d[i].T @ Quu @ k[i] + d[i].T @ Qux + Qu @ k[i]
            P = Qxx + k[i].T @ Quu @ k[i] + Qux.T @ k[i] + k[i].T @ Qux

        return k, d, Qu_list, Quu_list

    def forward(self, x, u, k, d, alpha, Qu_list, Quu_list):
        x_new = []
        u_new = []
        x_new.append(x[0])
        delta_J = 0.0
        for i in range(self.step):
            u_new.append(u[i] + k[i] @ (x_new[i] - x[i]) + alpha * d[i])
            x_new.append(self.vehicle_dynamic.RK3Function(x_new[i], u_new[i]))

            delta_J += alpha * (Qu_list[i] @ d[i]) + 0.5 * pow(alpha, 2) * (d[i].T @ Quu_list[i] @ d[i])
        delta_x_terminal = x_new[-1] - x[-1]
        delta_J += (delta_x_terminal.T @ self.cost_func.P_fun(self.ref_state[-1], x[-1]) @ delta_x_terminal + self.cost_func.p_fun(self.ref_state[-1], x[-1]) @ delta_x_terminal)
        J = self.evaluate(x_new, u_new)
        return x_new, u_new, J, delta_J

def regularized_persudo_inverse(mat, reg=1e-5):
    u, s, v = np.linalg.svd(mat)
    for i in range(len(s)):
        if s[i] < 0:
            s.at[i].set(0.0)
            print("Warning: inverse operator singularity{0}".format(i))
    diag_s_inv = np.diag(1. / (s + reg))
    return ca.DM(v.dot(diag_s_inv).dot(u.T))

def GetRsPathCost(path):
    path_cost = 0.0
    for length in path.lengths:
        if length >= 0.0:
            path_cost += length
        else:
            path_cost += abs(length)
    return path_cost

if __name__ == "__main__":
    start_pose = [30, 10, np.deg2rad(0.0)]
    goal_pose = [40, 7, np.deg2rad(0.0)]

    max_steer = 0.5
    wheel_base = 2.84
    step_size = 0.1
    max_curvature = np.tan(max_steer) / wheel_base
    rs_paths = calc_paths(start_pose[0], start_pose[1], start_pose[2], goal_pose[0], goal_pose[1], goal_pose[2], max_curvature, step_size)
    best_rs_path = None
    best_rs_cost = None

    for path in rs_paths:
        cost = GetRsPathCost(path)
        if not best_rs_cost or cost < best_rs_cost:
            best_rs_cost = cost
            best_rs_path = path
    u0 = [np.array([0.0, 0.0])] * (len(best_rs_path.x) - 1)
    x0 = [np.array([30, 10, 0.0 , 0.0])] * (len(best_rs_path.x) - 1)
    ref_path = np.vstack([best_rs_path.x, best_rs_path.y, best_rs_path.yaw, np.zeros(len(best_rs_path.x))]).T
    Q = np.diag((1, 1, 1, 0))
    R = np.eye(2)
    Q_Terminal = np.diag((1,1,1,0))*100
    print(Q_Terminal)
    print(len(best_rs_path.x))
    Vehicle = VehicleDynamics(wheel_base)
    Cost = CostFunc(ref_path, Q, R, Q_Terminal)

    solver = iLQR(Vehicle, Cost, ref_path)
    res = solver.solve(x0, u0)

    x_opt = res['x_hist'][-1]
    u_opt = res['u_hist'][-1]

    x = []
    y = []
    u = []
    for i in range(len(x_opt)):
        x.append(x_opt[i][0])
        y.append(x_opt[i][1])
    for i in range(len(x_opt)-1):
        u.append(u_opt[i].full().flatten())
    u = np.array(u)
    C = Cost.calc_cost(x0,u0)


    plt.plot(ref_path[:,0], ref_path[:,1])
    plt.plot(x,y)
    plt.show()

    x = [0, 100]
    y = [-0.5, -0.5]
    u = np.array(u)
    plt.plot(u[:, 0])
    plt.plot(u[:, 1])
    plt.plot(x, y)
    plt.show()

