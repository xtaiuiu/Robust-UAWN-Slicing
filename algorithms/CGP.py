# Use the method of "A novel projected gradient-like method for optimization problems with simple constraints"
# to optimize the problem:
# min f(x)
# s.t. x >= 0
import numpy as np
import scipy.linalg as lg

# minimize by using cvxpy for \sum\limits_{i} -a_i x_i ln(1 + b_i*y_i) + lam * (x^T y - p) + (x^T y - p)^2 /(2*rho),
# s.t., y >= q,
# the problem is transformed into:
# \sum\limits_{i} -a_i x_i ln(1 + b_i*(z_i + q_i)) + lam * (x^T (z+q)) - p) + (x^T (z+q) - p)^2 /(2*rho),
# s.t., z >= 0,
# So we minimize the variable z


def f(prob, lam, rho, x, z):
    """
    Evaluate the objective values at y
    :param prob: Problem object
    :param lam: lambda
    :param rho: rho
    :param x: x parameter
    :param z: current iteration point
    :return: func_val
    """
    z = z + prob.p_u
    xz_dot = np.sum(np.multiply(z, x))
    bz = np.multiply(prob.b, z)
    obj = np.sum(np.multiply(-prob.a * x, np.log1p(bz))) + lam * (xz_dot - prob.P) + np.square(xz_dot - prob.P) / (
                2 * rho)
    return obj


# the gradient of f
# g_i = -(a_i * b_i * x_i)/(1 + b_i * (z_i + q_i)) + lam * x_i + x_i*(x^T * (z + q) - p)/rho
def g(prob, lam, rho, x, z):
    # the parameters are the same as f
    z = z + prob.p_u
    xz_dot = np.sum(np.multiply(z, x))
    grad = np.array([- (prob.a[i]*prob.b[i]*x[i]) / (1 + prob.b[i]*z[i]) + lam * x[i] + x[i] * (xz_dot - prob.P) / rho for i in range(len(z))])
    return grad


# Algorithm 1 in the paper
def pg_alg(prob, lam, rho, x, z0, tau1=1e-6, tau2=1e-6, tau3=1e-4):
    """
    :param f: callable, the objective function
    :param g: callable, the gradient of the function
    :param x0: initial feasible
    :param tau1: precision
    :return: f_opt, x_opt
    """
    beta, sigma, gamma = 0.1, 0.9, 6
    alpha = gamma
    n, n_max = 0, 20000
    z = z0
    P = np.minimum(z, g(prob, lam, rho, x, z))  # P(x^k, \Nabla f(x^k))
    P1 = P
    f_old, f_cur = 1e8, f(prob, lam, rho, x, z)

    # helper function that can evaluate m_k that satisfies Armijo rule
    def Armijo_stepsize():
        m_max = 100
        for m in range(m_max):
            #print(f"0 = {np.zeros(len(z))}, xk_beta = {z - beta**m * gamma * P}")
            z_beta_gamma = np.maximum(np.zeros(len(z)), z - beta**m * gamma * P)  # x^k(beta^m gamma)
            lhs = f(prob, lam, rho, x, z) - f(prob, lam, rho, x, z_beta_gamma)  # LHS of Equ. (20)
            rhs = sigma * np.dot(P, z - z_beta_gamma)  # RHS of Equ. (20)
            if lhs >= rhs:
                break
        return m


    while (lg.norm(P) > tau2) and (lg.norm(P) > tau1*lg.norm(P1)) and (n < n_max) and abs((f_old - f_cur)/f_cur) > tau3:
        #print(f"n = {n}, precision = {lg.norm(P)}, f_old = {f_old: .6f}, f_cur = {f_cur: 6f},
        # g_norm = {lg.norm(g(prob, lam, rho, x, z))}, alpha = {alpha}")
        f_old = f_cur
        m = Armijo_stepsize()
        alpha = beta**m*gamma
        z = np.maximum(np.zeros(len(z)), z - alpha*P)
        P = np.minimum(z, g(prob, lam, rho, x, z))  # update P
        f_cur = f(prob, lam, rho, x, z)
        n += 1
    return f(prob, lam, rho, x, z), z+prob.p_u


