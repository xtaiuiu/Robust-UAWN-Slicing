# Solve MP-Relax　by using augmented Lagrange method, wherein the augmented Lagrange function
# is solved by the Sequential Least Squares Programming method
import time

import numpy as np
from scipy.optimize import minimize

from algorithms.MP_Relax_algorithms.main_algorithm.DAL.Benchmark_static_power import static_power_alloc
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import DAL_alg
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_p_BFGS, optimize_x_cvx

from utils.logger import get_logger

logger = get_logger(__name__)
def Lag_SQP_alg(prob, x_init, p_init, eps=1e-6):
    """
    :param prob: the optimization problem
    :param x_init: the initial x
    :param p_init: the initial p
    :return: f: the function value
    x, p, n: numpy arrays that are the optimal x, optimal p, the number of outer iterations of Lag_pgd_alg.
    """
    # check feasibility
    if prob.x_u@prob.c > prob.B_tot or prob.x_u@prob.p_u > prob.P:
        raise ValueError(f"!!! Lag_pgd_alg: infeasible problem.  !!! B_tot = {prob.B_tot}, "
                         f"c*x_u = {prob.x_u@prob.c}, P = {prob.P}, dot_pu_xu = "
                         f"{prob.x_u@prob.p_u}")
    # the main algorithm for my paper
    lam, rho = 1, 1  # initialize lam, rho
    lam_min, lam_max = -1e10, 1e10
    beta = 2.0
    f_prev = 1e8
    n, n_max = 0, 300
    x, p = x_init, p_init

    f_cur = prob.objective_function(x, p)
    logger.debug(f"AL_SQP: n = {n}, f_pre = {f_prev: .2f}, lam = {lam: .6f}, rho = {rho: .6f}, f_cur = {f_cur: .2f} "
          f"equ_constraint = {np.dot(x, p) - prob.P: .6f}, xp = {x @ p}")
    while n < n_max and abs((f_cur - f_prev)) > eps and abs((f_cur - f_prev)/f_cur) > eps:
    #while n < n_max and abs((f_prev - f_cur)/f_cur) > eps and abs(x@p - prob.P) > 0.1:
        f_prev = f_cur
        f_cur, x, p = solve_augmented_lagrange_cvxpy(prob, lam, rho, x, p)
        if abs(np.dot(x, p) - prob.P) > 1:
            lam += (np.dot(x, p) - prob.P) / rho
            lam = min(max(lam, lam_min), lam_max)
        rho /= beta
        n += 1
        logger.debug(f"AL_SQP: n = {n}, f_pre = {f_prev: .2f}, lam = {lam: .6f}, rho = {rho: .6f}, f_cur = {f_cur: .2f} "
            f"equ_constraint = {np.dot(x, p) - prob.P: .6f}, xp = {x @ p}")

    #print(f" DAL finished in n = {n}, f_pre = {f_prev: .8f}, f_cur = {f_cur: .8f}, constraint = {x@p - prob.P: .8f}")
    return prob.objective_function(x, p), x, p, n


def solve_augmented_lagrange_cvxpy(prob, lam, rho, x_init, p_init):
    """
    Solve the augmented Lagrangian problem using scipy.optimize.minimize with SLSQP
    
    Args:
        prob: OptimizationProblem object containing problem parameters
        lam: Lagrange multiplier (dual variable)
        rho: Penalty parameter
        x_init: Initial value for x
        p_init: Initial value for p
        
    Returns:
        tuple: (optimal_value, x_opt, p_opt) where:
            - optimal_value: The optimal objective value
            - x_opt: Optimal x values
            - p_opt: Optimal p values
    """
    N = len(prob.a)  # Number of users
    
    # Combine x and p into a single vector for optimization
    def unpack_vars(vars):
        x = vars[:N]
        p = vars[N:]
        return x, p
    
    # Define the objective function
    def objective(vars):
        x, p = unpack_vars(vars)
        obj = 0.0
        for n in range(N):
            obj += -prob.a[n] * x[n] * np.log(1 + prob.b[n] * p[n])
        
        # Add augmented Lagrangian terms
        constraint_violation = np.sum(x * p) - prob.P
        obj += lam * constraint_violation + (1 / (2 * rho)) * (constraint_violation ** 2)
        
        return obj
    
    # Define constraints
    def budget_constraint(vars):
        x = vars[:N]
        return prob.B_tot - np.dot(prob.c, x)
    
    # Bounds for x and p (x >= x_u, p >= p_u)
    bounds = ([(prob.x_u[i], None) for i in range(N)] +  # x bounds
              [(prob.p_u[i], None) for i in range(N)])   # p bounds
    
    # Initial guess
    x0 = np.concatenate([x_init, p_init])
    
    # Define constraints
    constraints = [
        {'type': 'ineq', 'fun': budget_constraint}  # c^T x ≤ B_tot
    ]
    
    # Solve the problem
    result = minimize(
        objective,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge. Status: {result.message}")
    
    # Extract and return the solution
    x_opt, p_opt = unpack_vars(result.x)
    return result.fun, x_opt, p_opt


if __name__ == '__main__':
    from algorithms.MP_Relax_algorithms.MP_Relax_problem import create_optimization_instance
    np.random.seed(0)
    prob = create_optimization_instance(n=50)
    t = time.perf_counter()
    f, x, y, n = Lag_SQP_alg(prob, prob.x_u, prob.p_u)
    print(f"finished in {time.perf_counter() - t} sec, f = {f: .8f}")

    t = time.perf_counter()
    # use static allocation as a warm start
    f_static, x_static, p_static = static_power_alloc(prob)
    f, x, y, _, _, _, _ = DAL_alg(prob, x_static, p_static, subx_optimizer=optimize_x_cvx,
                                  subp_optimizer=optimize_p_BFGS)
    print(f"finished in {time.perf_counter() - t} sec, f = {f: .8f}")


