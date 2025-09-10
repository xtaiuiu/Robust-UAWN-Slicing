# Solve MP-Relax　by using augmented Lagrange method, wherein the augmented Lagrange function
# is solved by the Gradient Projection Descent method
import time

import numpy as np
from scipy.optimize import minimize

from algorithms.MP_Relax_algorithms.main_algorithm.DAL.Benchmark_static_power import static_power_alloc
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import DAL_alg
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_p_BFGS, optimize_x_cvx


def Lag_GPD_alg(prob, x_init, p_init, eps=1e-4):
    """
    :param prob: the optimization problem
    :param x_init: the initial x
    :param p_init: the initial p
    :return: f: the function value
    x, p, n: numpy arrays that are the optimal x, optimal p, the number of outer iterations of Lag_pgd_alg.
    """
    # check feasibility
    if prob.x_u @ prob.c > prob.B_tot or prob.x_u @ prob.p_u > prob.P:
        raise ValueError(f"!!! Lag_pgd_alg: infeasible problem.  !!! \n B_tot = {prob.B_tot}, "
                         f"c*x_u = {prob.x_u @ prob.c}, P = {prob.P}, dot_pu_xu = "
                         f"{prob.x_u @ prob.p_u}")
    # the main algorithm for my paper
    lam, rho = 1, 1  # initialize lam, rho
    lam_min, lam_max = -1e10, 1e10
    beta = 2.0
    f_prev = 1e8
    n, n_max = 0, 300
    x, p = x_init, p_init

    f_cur = prob.objective_function(x, p)
    while n < n_max and abs((f_cur - f_prev)) > eps and abs((f_cur - f_prev) / f_cur) > eps:
        # while n < n_max and abs((f_prev - f_cur)/f_cur) > eps and abs(x@p - prob.P) > 0.1:
        print(f"AL_GPD: n = {n}, f_pre = {f_prev: .2f}, lam = {lam: .6f}, rho = {rho: .6f}, f_cur = {f_cur: .2f} "
              f"equ_constraint = {np.dot(x, p) - prob.P: .6f}")
        f_prev = f_cur
        f_cur, x, p = solve_augmented_lagrange_GPD(prob, lam, rho, x, p)
        lam += (np.dot(x, p) - prob.P) / rho
        lam = min(max(lam, lam_min), lam_max)
        rho /= beta
        n += 1
    # print(f" DAL finished in n = {n}, f_pre = {f_prev: .8f}, f_cur = {f_cur: .8f}, constraint = {x@p - prob.P: .8f}")
    return prob.objective_function(x, p), x, p, n


def solve_augmented_lagrange_GPD(prob, lam, rho, x_init, p_init, max_iter=1000, tol=1e-6, alpha=0.1, beta=0.5, momentum=0.9):
    """
    Solve the augmented Lagrangian problem using Gradient Projection Descent (GPD)

    Args:
        prob: OptimizationProblem object containing problem parameters
        lam: Lagrange multiplier (dual variable)
        rho: Penalty parameter
        x_init: Initial value for x
        p_init: Initial value for p
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        alpha: Initial step size
        beta: Backtracking line search parameter

    Returns:
        tuple: (optimal_value, x_opt, p_opt)
            - optimal_value: The optimal objective value
            - x_opt: Optimal x values
            - p_opt: Optimal p values
    """
    from utils.projction import proj_generalized_simplex_lb

    N = len(prob.a)
    x = x_init.copy()
    p = p_init.copy()

    def compute_objective(x, p):
        """Compute the augmented Lagrangian objective"""
        obj = -np.sum(prob.a * x * np.log(1 + prob.b * p))
        constraint_violation = np.dot(x, p) - prob.P
        obj += lam * constraint_violation + (1 / (2 * rho)) * (constraint_violation ** 2)
        return obj

    def compute_gradients(x, p):
        """Compute gradients with respect to x and p"""
        # Gradient of the objective part
        grad_x_obj = -prob.a * np.log(1 + prob.b * p)
        grad_p_obj = -prob.a * x * (prob.b / (1 + prob.b * p))

        # Gradient of the augmented Lagrangian terms
        constraint_violation = np.dot(x, p) - prob.P
        grad_x_lag = lam * p + (1 / rho) * constraint_violation * p
        grad_p_lag = lam * x + (1 / rho) * constraint_violation * x

        # Total gradients
        grad_x = grad_x_obj + grad_x_lag
        grad_p = grad_p_obj + grad_p_lag

        return grad_x, grad_p

    # Initialize momentum terms
    vx = np.zeros_like(x_init)
    vp = np.zeros_like(p_init)
    prev_obj = np.inf
    restart_counter = 0
    max_restarts = 5

    # Main GPD loop with Nesterov momentum
    for it in range(max_iter):
        # Compute current objective and gradients
        current_obj = compute_objective(x, p)

        # Check for objective increase (for adaptive restart)
        if current_obj > prev_obj and restart_counter < max_restarts:
            # Restart momentum
            vx = np.zeros_like(x)
            vp = np.zeros_like(p)
            restart_counter += 1
            if restart_counter == max_restarts:
                print("Max restarts reached, continuing without momentum")
        prev_obj = current_obj

        # Compute lookahead position for NAG
        x_lookahead = x + momentum * vx
        p_lookahead = p + momentum * vp

        # Project lookahead positions
        x_lookahead = proj_generalized_simplex_lb(x_lookahead, prob.c, prob.B_tot, prob.x_u)
        p_lookahead = np.maximum(p_lookahead, prob.p_u)

        # Compute gradient at lookahead position
        grad_x, grad_p = compute_gradients(x_lookahead, p_lookahead)

        # Update with momentum
        vx = momentum * vx - alpha * grad_x
        vp = momentum * vp - alpha * grad_p

        # Take step
        x_new = x + vx
        p_new = p + vp

        # Project back to feasible set
        x_new = proj_generalized_simplex_lb(x_new, prob.c, prob.B_tot, prob.x_u)
        p_new = np.maximum(p_new, prob.p_u)

        # Line search with adaptive step size
        step = 1.0
        new_obj = compute_objective(x_new, p_new)

        # Backtracking line search with Armijo condition
        while (new_obj > current_obj - 0.5 * step * (np.sum(grad_x ** 2) + np.sum(grad_p ** 2)) and
               step > 1e-10):
            step *= beta
            vx = momentum * vx * beta
            vp = momentum * vp * beta
            x_new = x + vx
            p_new = p + vp
            x_new = proj_generalized_simplex_lb(prob.c, x_new, prob.B_tot, prob.x_u)
            p_new = np.maximum(p_new, prob.p_u)
            new_obj = compute_objective(x_new, p_new)
        print(f"it = {it: 5d}, obj = {new_obj:.10f}, constraint violation = {np.dot(x_new, p_new) - prob.P:.10f}")
        # Check convergence using optimality condition for projected gradient descent
        # Compute gradient at current point
        grad_x, grad_p = compute_gradients(x, p)
        
        # Take gradient step and project
        x_grad_step = x - alpha * grad_x
        p_grad_step = p - alpha * grad_p
        x_proj = proj_generalized_simplex_lb(prob.c, x_grad_step, prob.B_tot, prob.x_u)
        p_proj = np.maximum(p_grad_step, prob.p_u)
        
        # Check if the projected gradient step is close to current point
        if (np.linalg.norm(x_proj - x) < tol and 
            np.linalg.norm(p_proj - p) < tol):
            break

        x, p = x_new, p_new

    # Compute final objective value
    final_obj = compute_objective(x, p)
    return final_obj, x, p


if __name__ == '__main__':
    from algorithms.MP_Relax_algorithms.MP_Relax_problem import create_optimization_instance

    np.random.seed(0)
    prob = create_optimization_instance(n=40)
    t = time.perf_counter()
    f, x, y, n = Lag_GPD_alg(prob, prob.x_u, prob.p_u)
    print(f"finished in {time.perf_counter() - t} sec, f = {f: .8f}")

    t = time.perf_counter()
    # use static allocation as a warm start
    f_static, x_static, p_static = static_power_alloc(prob)
    f, x, y, _, _, _, _ = DAL_alg(prob, x_static, p_static, subx_optimizer=optimize_x_cvx,
                                  subp_optimizer=optimize_p_BFGS)
    print(f"finished in {time.perf_counter() - t} sec, f = {f: .8f}")


