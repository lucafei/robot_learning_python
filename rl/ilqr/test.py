import casadi as ca
import numpy as np

# Parameters (example values, adjust as needed)
N = 20  # Prediction horizon
dt = 0.1  # Time step

# Opti instance
opti = ca.Opti()

# Variables
s = opti.variable(N+1)  # Position
v = opti.variable(N+1)  # Speed
a = opti.variable(N)    # Acceleration

# Compute jerk as an MX vector (not a Python list)
jerk = ca.MX(N-1, 1)  # Create an MX vector of size N-1
for k in range(N-1):
    jerk[k] = (a[k+1] - a[k]) / dt  # Jerk = da/dt

# Objective function: Minimize sum of squared acceleration and jerk
objective = ca.sumsqr(a) + 0.1 * ca.sumsqr(jerk)  # Use ca.sumsqr directly
opti.minimize(objective)

# Add other constraints (e.g., dynamics, safety, bounds) as in your original code
# Example dynamic constraints
for k in range(N):
    opti.subject_to(s[k+1] == s[k] + v[k] * dt)
    opti.subject_to(v[k+1] == v[k] + a[k] * dt)

# Example boundary constraints
opti.subject_to(s[0] == 0)
opti.subject_to(v[0] == 10)
opti.subject_to(v >= 0)
opti.subject_to(v <= 20)
opti.subject_to(a >= -3)
opti.subject_to(a <= 3)
for k in range(N-1):
    opti.subject_to((a[k+1] - a[k])/dt <= 2)  # Jerk constraint
    opti.subject_to((a[k+1] - a[k])/dt >= -2)

# Solver
opti.solver('ipopt')
sol = opti.solve()

# Extract results
s_opt = sol.value(s)
v_opt = sol.value(v)
a_opt = sol.value(a)

# Plotting (save to file to avoid GUI issues)
import matplotlib.pyplot as plt
t = np.linspace(0, N*dt, N+1)
plt.plot(t, v_opt, label='Speed (m/s)')
plt.plot(t[:-1], a_opt, label='Acceleration (m/s^2)')
plt.legend()
plt.grid(True)
plt.savefig('speed_plan.png')
plt.close()