import cvxpy as cp
import numpy as np
from cvxpy import SolverError
from scipy.optimize import fmin_l_bfgs_b, minimize
import copy


# Optimize Sub_x by cvxpy
# minimize -x^T nu + lam*(x^p - P) + 1/(2*rho)* \norm{x^p - P}^2
# s.t. x >= x_u,
# c^T x <= B_tot
def optimize_x_cvx(prob, lam, rho, p, x0, eps=1e-2):
    """
    :param prob: an OptimizationProblem object
    :param lam: lambda
    :param rho: rho
    :param p: the p variable, which is regarded as an array
    :return:
    """
    # check feasibility of the problem:
    if np.dot(prob.c, prob.x_u) + 1e-4 > prob.B_tot:
        raise ValueError("optimize_x_cvx: infeasible problem Sub_x")
    n = len(prob.a)
    x = cp.Variable(shape=n)
    nu = np.array([prob.a[i] * np.log1p(prob.b[i] * p[i]) for i in range(n)])
    xp_dot = cp.sum(cp.multiply(p, x))
    obj = cp.sum(cp.multiply(-nu, x)) + lam * (xp_dot - prob.P) + cp.square(xp_dot - prob.P) / (2 * rho)
    constraints = [x >= prob.x_u, prob.c @ x <= prob.B_tot]
    problem = cp.Problem(cp.Minimize(obj), constraints)
    # problem.solve(solver=cp.CLARABEL)

    try:
        problem.solve(solver=cp.MOSEK, verbose=False)
    except SolverError:
        print(f"Problem was not solved to optimality. Status: {problem.status}")
        problem.solve(solver=cp.CLARABEL, verbose=False)

    # test feasibility
    if min(x.value - prob.x_u) + eps < 0 or prob.c@x.value > prob.B_tot + eps:
        raise ValueError(f"cvx solution infeasible: min(x.value - prob.x_u) = {min(x.value - prob.x_u)}, "
                         f"c^T x - B_tot = {prob.c@x.value - prob.B_tot}, f_cvx = {problem.value: .8f}")
    return problem.value, x.value


def optimize_x_scipy(prob, lam, rho, p):
    """
    :param prob: an OptimizationProblem object
    :param lam: lambda
    :param rho: rho
    :param p: the p variable, which is regarded as an array
    :return:
    """
    nu = np.array([prob.a[i] * np.log1p(prob.b[i] * p[i]) for i in range(len(prob.a))])
    objective = lambda x: -np.dot(x, nu) + lam * (np.dot(x, p) - prob.P) + (np.dot(x, p) - prob.P) ** 2 / (rho * 2)
    linear_constraint = {'type': 'ineq', 'fun': lambda x: prob.B_tot - np.dot(prob.c, x)}
    x0 = copy.deepcopy(prob.x_u)
    x0[-1] = (prob.B_tot - np.dot(x0[:-1], prob.c[:-1])) / prob.c[-1]
    result = minimize(objective, x0, method='trust-constr', constraints=[linear_constraint],
                      bounds=[(x_ui, None) for x_ui in prob.x_u])
    return result.fun, result.x


# Optimize Sub_p by cvxpy
# minimize by using cvxpy for
# \min_p \sum\limits_{i} -omega_i ln(1 + b_i*p_i) + lam * (x^T y - P) + (x^T p - P)^2 /(2*rho),
# s.t., p >= p_u,
def optimize_p_cvx(prob, lam, rho, x):
    """
    :param prob: an OptimizationProblem object
    :return: opt_val => double, opt_val => numpy array
    """
    n = len(prob.a)
    p = cp.Variable(shape=n)
    xp_dot = cp.sum(cp.multiply(p, x))
    bp = cp.multiply(prob.b, p)
    obj = cp.sum(cp.multiply(-prob.a * x, cp.log1p(bp))) + lam * (xp_dot - prob.P) + cp.square(xp_dot - prob.P) / (
                2 * rho)
    constraints = [p >= prob.p_u]
    prob_cvx = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob_cvx.solve(solver=cp.CLARABEL)
        if prob_cvx.status == cp.OPTIMAL:  # 检查问题是否成功求解
            assert np.sum(p.value) > 0, "p.value is not positive"
        else:
            print(f"!!!!!!!!!!!!!!!!!! solver failed with status: {prob_cvx.status} !!!!!!!!!!!!!!!!!!")
    except cp.error.SolverError as e:
        print(f"!!!!!!!!!!!!!!!!!! solver failed: {e} !!!!!!!!!!!!!!!!!!")
        prob_cvx.solve(solver=cp.MOSEK)
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!! unexpected error: {e} !!!!!!!!!!!!!!!!!!")
        prob_cvx.solve(solver=cp.CLARABEL)
    # prob.solve()
    # check feasibility

    return prob_cvx.value, p.value


# Optimize Sub_p by L-BFGS-B
# minimize by using cvxpy for
# \min_p \sum\limits_{i} -omega_i ln(1 + b_i*p_i) + lam * (x^T y - P) + (x^T p - P)^2 /(2*rho),
# s.t., p >= p_u,
def optimize_p_BFGS(prob, lam, rho, x):
    # Define the objective function at point p
    def BFGS_objective(p):
        return np.sum(-prob.a * x * np.log1p(prob.b * p)) + lam * (x @ p - prob.P) + (x @ p - prob.P) ** 2 / (2 * rho)

    # Define the gradient at point p
    def BFGS_gradient(p):
        t = (lam + (x @ p - prob.P) / rho) * x
        return np.array([-prob.a[i] * x[i] * prob.b[i] / (1 + prob.b[i] * p[i]) + t[i] for i in range(len(x))])

    # Initial guess
    p0 = prob.p_u
    bounds = [(pu_i, None) for pu_i in prob.p_u]

    # Perform the minimization using L-BFGS-B
    result = minimize(BFGS_objective, p0, method='L-BFGS-B', jac=BFGS_gradient, bounds=bounds,
                      options={'maxcor': 200, 'maxls': 50, 'maxiter': 10000, 'ftol': 1e-4})

    if not result.success:
        raise Exception(f"L-BFGS-B failed: {result.message}")

    # check feasibility:
    if np.min(result.x - prob.p_u) + 1e-6 < 0:
        raise ValueError(f"L-BFGS-B solution error: np.min(result.x - prob.p_u) = {np.min(result.x - prob.p_u)}")
    # Print the result
    return result.fun, result.x
