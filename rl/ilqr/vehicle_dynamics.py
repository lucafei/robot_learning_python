import numpy as np
import casadi as ca

class VehicleDynamics:
    def __init__(self, wheelbase):
        self.wheelbase = wheelbase
        self.make_variable()
        self.BicycleModelRk3()
        self.Jacobian()
        self.Hessian()


    def make_variable(self):
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        v = ca.SX.sym('v')
        steering = ca.SX.sym('steering')
        a = ca.SX.sym('a')
        steering_rate = ca.SX.sym('steering_rate')
        self.t = 0.1
        self.state = ca.vertcat(x, y, theta, v)
        self.control = ca.vertcat(steering, a)
        beta = ca.atan(2*ca.tan(steering)/self.wheelbase)
        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), v / self.wheelbase * ca.tan(steering), a)
        self.Dynamic_Func = ca.Function('f', [self.state, self.control], [rhs])

    # runge-kutta 3rd order integration for bicycle model
    def BicycleModelRk3(self):
        k1 = self.Dynamic_Func(self.state, self.control) * self.t
        k2 = self.Dynamic_Func(self.state + k1/2, self.control) * self.t
        k3 = self.Dynamic_Func(self.state - k1 + 2 * k2, self.control) * self.t
        self.RK3 = self.state + (k1 + 4 * k2 + k3) / 6
        self.RK3Function = ca.Function('RK3', [self.state, self.control], [self.RK3])


    def Jacobian(self):
        self.dfdx = ca.jacobian(self.RK3, self.state)
        self.dfdx_func = ca.Function("dfdx", [self.state, self.control], [self.dfdx])
        self.dfdu = ca.jacobian(self.RK3, self.control)
        self.dfdu_func = ca.Function("dfdu", [self.state, self.control], [self.dfdu])


    def Hessian(self):
        self.dfddx = ca.jacobian(self.dfdx, self.state)
        self.dfdxdu = ca.jacobian(self.dfdx, self.control)
        self.dfddu = ca.jacobian(self.dfdu, self.control)