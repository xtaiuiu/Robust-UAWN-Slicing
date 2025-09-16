# Optimize the AL problem with parameter lamda and rho by using 2-BCD algorithm.
# \min_{x, p} \sum\limits_{i} -a_i*x_i ln(1 + b_i*p_i) + lam * (x^T y - P) + (x^T p - P)^2 /(2*rho),
# s.t., x >= x_u,
# c^T x = B_tot
# p >= p_u,
import time

import numpy as np
from scipy.optimize import linprog

from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.GPM_algorithm import gpm_x
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_x_cvx, optimize_p_cvx, \
    optimize_x_scipy
from utils.logger import get_logger

logger = get_logger(__name__)

def bcd_al(prob, lam, rho, x_init, p_init, subx_optimizer=optimize_x_cvx,
           subp_optimizer=optimize_p_cvx, x_first=True, eps=1e-8):
    """
    :param prob: the original problem
    :param lam: lambda
    :param rho: rho
    :param x_first: whether x is optimized first
    :return: f, x, y
    """
    x, p = x_init, p_init  # an initial feasible point
    f_old = 1e8
    f = prob.Lagrangian_value(x, p, lam, rho)
    n, n_max = 0, 50000
    n_subx, n_subp = 0, 0  # the number of the solved sub-problems Sub_x and Sub_p, respectively
    while abs((f - f_old)/f) > eps and abs(f - f_old) > eps and n < n_max:
        # if np.linalg.norm(prob.g_x(x, p, lam, rho)) < 0.1:
        #     print(f" norm g = {np.linalg.norm(prob.g_x(x, p, lam, rho)): 6f}")
        #     break
        if x_first:
            t1 = time.perf_counter()
            f_old = f
            f, x = subx_optimizer(prob, lam, rho, p, x)
            n_subx += 1
            if abs(f - f_old) < eps or abs((f - f_old) / f) < eps:
                break
            logger.info(
                f"n = {n}, optimize x: lam = {lam: .4f}, rho = {rho: .8f}, f_old = {f_old: .8f}, f = {f: .8f}, time = {time.perf_counter() - t1: .8f}")
            t1 = time.perf_counter()
            f_old = f
            f, p = subp_optimizer(prob, lam, rho, x)
            n_subp += 1
            if abs(f - f_old) < eps or abs((f - f_old) / f) < eps:
                break
            logger.info(
                f"n = {n}, optimize p: lam = {lam :.4f}, rho = {rho: .8f}, f_old = {f_old:.8f}, f = {f:.8f}, time = {time.perf_counter() - t1: .8f}")
        else:
            t1 = time.perf_counter()
            f_old = f
            f, p = subp_optimizer(prob, lam, rho, x)
            n_subp += 1
            if abs(f - f_old) < eps or abs((f - f_old) / f) < eps:
                break
            logger.info(
                f"n = {n}, optimize p: lam = {lam :.4f}, rho = {rho: .8f}, f_old = {f_old:.8f}, f = {f:.8f}, time = {time.perf_counter() - t1: .8f}")

            t1 = time.perf_counter()
            f_old = f
            f, x = subx_optimizer(prob, lam, rho, p, x)
            n_subx += 1
            if abs(f - f_old) < eps or abs((f - f_old) / f) < eps:
                break
            logger.info(
                f"n = {n}, optimize x: lam = {lam: .4f}, rho = {rho: .8f}, f_old = {f_old: .8f}, f = {f: .8f}, time = {time.perf_counter() - t1: .8f}")

        n += 1
    #print(f"BCD finished in n = {n}, abs(f - f_old) = {abs(f - f_old): 12f}")
    return f, x, p, n_subx, n_subp
