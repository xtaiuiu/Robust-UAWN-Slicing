# solve the MP-Relax problem by using alternative optimization method,
# which optimize variables x and p alternatively. This algorithm is generally not converged
################ !!!!!!!! The result of AO is unstable.  Not Used !!!!!!!! ################

import time

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize, Bounds, LinearConstraint

from algorithms.MP_Relax_algorithms.main_algorithm.DAL.Benchmark_static_power import static_power_alloc


def optimize_x_given_p(prob, p_fixed):
    """Optimize x given fixed p using CVXPY."""
    N = len(prob.a)
    x = cp.Variable(N, nonneg=True)

    # Objective: maximize sum(a_n * x_n * log(1 + b_n * p_n))
    objective = cp.Maximize(cp.sum(cp.multiply(prob.a, x) * cp.log(1 + cp.multiply(prob.b, p_fixed))))

    # Constraints
    constraints = [
        x >= prob.x_u,  # x ≥ x_lower
        prob.c @ x <= prob.B_tot,  # c^T x ≤ B_tot
        x @ p_fixed <= prob.P  # x^T p ≤ P
    ]

    # Solve
    prob_cvx = cp.Problem(objective, constraints)
    prob_cvx.solve(solver=cp.ECOS)

    if prob_cvx.status != 'optimal':
        print(f"Warning: x optimization status: {prob_cvx.status}")

    return x.value


def optimize_p_given_x(prob, x_fixed):
    """Optimize p given fixed x using CVXPY."""
    N = len(prob.a)
    p = cp.Variable(N, nonneg=True)

    # Objective: maximize sum(a_n * x_n * log(1 + b_n * p_n))
    objective = cp.Maximize(cp.sum(cp.multiply(cp.multiply(prob.a, x_fixed), cp.log(1 + cp.multiply(prob.b, p)))))

    # Constraints
    constraints = [
        p >= prob.p_u,  # p ≥ p_lower
        x_fixed @ p <= prob.P  # x^T p ≤ P
    ]

    # Solve
    prob_cvx = cp.Problem(objective, constraints)
    prob_cvx.solve(solver=cp.ECOS)

    if prob_cvx.status != 'optimal':
        print(f"Warning: p optimization status: {prob_cvx.status}")

    return p.value


def alternative_optimization(prob, x_init, p_init, max_iter=100, tol=1e-6):
    """
    Solve the MP-Relax problem using alternative optimization.

    Args:
        prob: OptimizationProblem object containing problem parameters
        x_init: Initial value for x
        p_init: Initial value for p
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        tuple: (optimal_value, x_opt, p_opt, n_iters) where:
            - optimal_value: The optimal objective value
            - x_opt: Optimal x values
            - p_opt: Optimal p values
            - n_iters: Number of iterations performed
    """
    x = x_init.copy()
    p = p_init.copy()
    prev_obj = -np.inf
    n_iters = 0

    for n_iters in range(1, max_iter + 1):
        # Optimize p given x
        p_new = optimize_p_given_x(prob, x)
        if p_new is None:
            break

        # Optimize x given p
        x_new = optimize_x_given_p(prob, p_new)
        if x_new is None:
            break

        # Calculate current objective
        curr_obj = np.sum(prob.a * x_new * np.log(1 + prob.b * p_new))

        # Check convergence
        if np.abs(curr_obj - prev_obj) < tol * (1 + np.abs(prev_obj)):
            break

        # Update for next iteration
        x, p = x_new, p_new
        prev_obj = curr_obj

        # Print progress
        print(f"Iter {n_iters}: obj = {curr_obj:.6f}, "
              f"constraint violation = {np.dot(x, p) - prob.P:.6f}")

    # Calculate final objective value
    final_obj = np.sum(prob.a * x * np.log(1 + prob.b * p))
    return final_obj, x, p, n_iters


if __name__ == '__main__':
    from algorithms.MP_Relax_algorithms.MP_Relax_problem import create_optimization_instance

    np.random.seed(0)
    prob = create_optimization_instance(n=40)  # or your desired number of users
    t = time.perf_counter()
    # use static allocation as a warm start
    f_static, x_static, p_static = static_power_alloc(prob)

    # obj, x_opt, p_opt, n_iters = alternative_optimization(prob, prob.x_u, prob.p_u)
    obj, x_opt, p_opt, n_iters = alternative_optimization(prob, x_static, p_static)
    print(f"Optimal value: {obj:.6f}, iterations: {n_iters}")
