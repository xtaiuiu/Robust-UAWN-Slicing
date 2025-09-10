import numpy as np
from scipy.optimize import minimize

# Define a non-convex smooth objective function (e.g., Rosenbrock function)
def objective(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2  # Example: Rosenbrock function

# Define the gradient of the objective (optional for better performance)
def grad_objective(x):
    grad = np.zeros_like(x)
    grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

# Define the constraints: x >= 0 and a linear constraint c^T x <= B
def constraint1(x):
    return x[0]  # x[0] >= 0

def constraint2(x):
    return x[1]  # x[1] >= 0

def constraint3(x, c, B):
    return B - np.dot(c, x)  # c^T x <= B

# Set up the problem data
c = np.array([1, 2])
B = 10
x0 = np.array([2, 2])  # Initial guess

# Define constraints
constraints = [
    {'type': 'ineq', 'fun': constraint1},
    {'type': 'ineq', 'fun': constraint2},
    {'type': 'ineq', 'fun': constraint3, 'args': (c, B)}
]

# Solve the problem using the 'trust-constr' method
result = minimize(
    objective,
    x0,
    jac=grad_objective,  # Optional: Provide the gradient if available
    method='trust-constr',  # Trust-region method for non-convex problems
    constraints=constraints,
    options={'disp': True}  # Display optimization progress
)

# Check results
if result.success:
    print("Optimal solution:", result.x)
    print("Objective function value:", result.fun)
else:
    print("Optimization failed:", result.message)
