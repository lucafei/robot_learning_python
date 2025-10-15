import casadi as ca
from sklearn.neighbors import radius_neighbors_graph


class CostFunc:
    def __init__(self, ref_path, Q, R, Q_terminal, SoftConstrain = []):
        self.ref_path = ref_path
        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal
        self.Constrain =  SoftConstrain
        self.SoftConstrain = None
        self.state_ref =None
        self.state = None
        self.control = None
        self.StageCost =None
        self.TerminalCost = None
        self.StageCostFunc =None
        self.TerminalCostFunc = None
        self.StageCostFunction = None
        self.lu = None
        self.lx = None
        self.luu_fun = None
        self.lu_fun = None
        self.lxx_fun = None
        self.lx_fun = None
        self.lux_fun = None
        self.p_fun = None
        self.P_fun = None
        self.soft_constrain_formular()
        self.state_cost()
        self.calc_jacobian()
        self.calc_hessian()
        self.calc_terminal_func()

    def soft_constrain_formular(self):
        vio = ca.SX.sym("vio")
        self.SoftConstrain = ca.Function("constrain", [vio],[0.01 * ca.exp(-10 * vio)])
    def state_cost(self):
        x_ref = ca.SX.sym('x_ref')
        y_ref = ca.SX.sym('y_ref')
        yaw_ref = ca.SX.sym('yaw_ref')
        v_ref = ca.SX.sym('v_ref')
        self.state_ref = ca.vertcat(x_ref, y_ref, yaw_ref, v_ref)

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        v = ca.SX.sym('v')
        steering = ca.SX.sym('steering')
        a = ca.SX.sym('a')
        self.state = ca.vertcat(x,y,theta,v)
        self.control = ca.vertcat(steering, a)

        state_error = self.state - self.state_ref
        self.StageCost = state_error.T @ self.Q @ state_error
        self.StageCost += self.control.T @ self.R @ self.control
        for con in self.Constrain:
            self.StageCost += self.SoftConstrain(con)

        self.TerminalCost = state_error.T @ self.Q_terminal @ state_error
        self.StageCostFunc = ca.Function("stage_cost", [self.state_ref, self.state, self.control], [self.StageCost])
        self.TerminalCostFunc = ca.Function("TerminalCost", [self.state_ref, self.state], [self.TerminalCost])

    def calc_cost(self, State, Control):
        cost = 0
        self.StageCostFunction = 0
        for i in range( len(self.ref_path) - 1):
            cost += self.StageCostFunc(self.ref_path[i], State[i], Control[i])
            self.StageCostFunction += self.StageCostFunc(self.ref_path[i],self.state,self.control)
        cost += self.TerminalCostFunc(self.ref_path[-1], State[-1])
        return cost

    def calc_jacobian(self):
        self.lx = ca.jacobian(self.StageCost, self.state)
        self.lu = ca.jacobian(self.StageCost, self.control)
        self.lx_fun = ca.Function("lx", [self.state_ref, self.state, self.control], [self.lx])
        self.lu_fun = ca.Function("lu", [self.state_ref, self.state, self.control], [self.lu])

    def calc_hessian(self):
        self.lxx_fun = ca.Function("lxx", [self.state_ref, self.state, self.control], [ca.jacobian(self.lx, self.state)])
        self.lux_fun = ca.Function("lux", [self.state_ref, self.state, self.control], [ca.jacobian(self.lu, self.state)])
        self.luu_fun = ca.Function("luu", [self.state_ref, self.state, self.control], [ca.jacobian(self.lu, self.control)])

    def calc_terminal_func(self):
        p = ca.jacobian(self.TerminalCost, self.state)
        self.p_fun = ca.Function("p", [self.state_ref, self.state], [p])
        self.P_fun = ca.Function("P", [self.state_ref, self.state], [ca.jacobian(p, self.state)])
