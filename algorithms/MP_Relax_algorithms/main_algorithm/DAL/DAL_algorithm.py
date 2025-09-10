# my main algorithm, i.e., DAL
import time

import numpy as np

from algorithms.MP_Relax_algorithms.main_algorithm.BCD.BCD_algorithm import bcd_al
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_x_cvx, optimize_p_cvx, \
    optimize_p_BFGS
from utils.logger import get_logger

logger = get_logger(__name__)
# solve the problem presented at the top by penalty Lagrangian method with BCD
# lam and rho are updated according to Algorithm 2 in "PENALTY DUAL DECOMPOSITION METHOD FOR
# NONSMOOTH NONCONVEX OPTIMIZATION—PART I"
def DAL_alg(prob, x_init, p_init, subx_optimizer=optimize_x_cvx, subp_optimizer=optimize_p_cvx, eps=1e-4):
    """
    :param prob:
    :param x_init:
    :param p_init:
    :param subx_optimizer:
    :param subp_optimizer:
    :return: f: the function value, n: int, the number of outer iterations of DAL
    x, p, n, inner_x, inner_p: numpy arrays that are the optimal x, optimal p, the number of Sub_x and Sub_p solved.
    BCD_times: numpy that has length n. Each element represents the BCD_time during a single outer iteration
    """
    # check feasibility
    if prob.x_u@prob.c > prob.B_tot or prob.x_u@prob.p_u > prob.P:
        raise ValueError(f"!!! DAL: infeasible problem.  !!! \n B_tot = {prob.B_tot}, "
                         f"c*x_u = {prob.x_u@prob.c}, P = {prob.P}, dot_pu_xu = "
                         f"{prob.x_u@prob.p_u}")
    # the main algorithm for my paper
    lam, rho = 1, 1  # initialize lam, rho
    lam_min, lam_max = -1e10, 1e10
    beta = 2.0
    f_prev = 1e8
    n, n_max = 0, 20
    x, p = x_init, p_init
    n_inner_x, n_inner_p = [], []  # the number of inner iterations
    BCD_times = []  # the time spend in BCD

    f_cur = prob.objective_function(x, p)
    while n < n_max and abs((f_cur - f_prev)) > eps and abs((f_cur - f_prev)/f_prev) > eps:
    #while n < n_max and abs((f_prev - f_cur)/f_cur) > eps and abs(x@p - prob.P) > 0.1:
        logger.debug(f"DAL: n = {n}, f_pre = {f_prev: .2f}, lam = {lam: .6f}, rho = {rho: .6f}, "
                f"f_cur = {f_cur: .2f} equ_constraint = {np.dot(x, p) - prob.P: .6f}, xp = {x @ p}")
        f_prev = f_cur
        t1 = time.perf_counter()
        f_cur, x, p, n_x, n_p = bcd_al(prob, lam, rho, x, p, subx_optimizer, subp_optimizer, x_first=False, eps=eps)
        BCD_times.append(round(time.perf_counter() - t1, 8))
        n_inner_x.append(n_x)
        n_inner_p.append(n_p)
        if abs(np.dot(x, p) - prob.P) > 1:
            lam += (np.dot(x, p) - prob.P) / rho
            lam = min(max(lam, lam_min), lam_max)
        rho /= beta
        n += 1
    return prob.objective_function(x, p), x, p, n, np.array(n_inner_x), np.array(n_inner_p), np.array(BCD_times)


