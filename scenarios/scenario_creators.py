import inspect
import math
import pickle
import time

from algorithms.MP_Relax_algorithms.MP_Relax_problem import MpRelaxProblem
from algorithms.MP_Relax_algorithms.benchmark_algorithms.Lagrange_SQP import Lag_SQP_alg
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.Benchmark_static_power import static_power_alloc
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import DAL_alg
from network_classes.physical_network import PhysicalNetwork
from network_classes.scenario import Scenario
from network_classes.uav import Uav
from scenarios.slice_creators import create_slice_set
import numpy as np

from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.GPM_algorithm import gpm_x


def create_scenario(n_slices, network_radius, p_max=10, b_tot=50, n_UEs_per_slice=10):
    # a scenario consists of: an UAV, a physical network (mainly physical parameters), a set of slices

    #(self, uav, p_max, b_tot, radius, h_min, h_max, theta_min, theta_max, t_s, g_0, alpha):
    #(self, height, height_prev, theta, speed):
    # uav parameters
    uav_height = 10
    uav_theta = np.pi/3
    uav_speed = 5
    uav_height_prev = 10
    uav_x, uav_y = 50, 0
    uav_x_bar, uav_y_bar = np.random.uniform(-network_radius, network_radius, 2)
    uav = Uav(uav_height, uav_height_prev, uav_theta, uav_speed, uav_x, uav_y, uav_x_bar, uav_y_bar)

    # network parameters
    h_min = 10
    h_max = 150
    theta_min = np.pi/10
    theta_max = np.pi/2.5
    t_s = 200
    #g_0 = 5
    g_0 = 3.24e-4
    alpha = 2
    N_0 = 4.0e-21  # background noise power density, in W/Hz, which is -174 dBm/Hz.
    sigma = 1e6 * N_0  # background noise power, we consider a maximum bandwidth of 1 MHz
    pn = PhysicalNetwork(uav, p_max, b_tot, network_radius, h_min, h_max, theta_min, theta_max, t_s, g_0, alpha, sigma)
    slices = create_slice_set(n_slices, network_radius, n_UEs_per_slice)
    scenario = Scenario(pn, uav, slices)
    return scenario

def load_scenario(filename):
    with open(filename, 'rb') as file:
        sc = pickle.load(file)
    return sc


def save_scenario(sc, filename):
    with open(filename, 'wb') as file:
        pickle.dump(sc, file)


def scenario_to_problem(sc):
    """
    Create an OptimizationProblem object from a scenario
    :param sc: the scenario
    :return: an OptimizationProblem object
    """

    # First, optimize the UAV height using Bender's decomposition
    uav, pn = sc.uav, sc.pn
    norm_phi_u = np.linalg.norm(np.array([uav.x, uav.y]))  # norm of \phi_u, i.e., the distance between the UAV and the origin
    omega_u = pn.radius + norm_phi_u
    d_u = np.linalg.norm(np.array([uav.x, uav.y]) - np.array([uav.x_bar, uav.y_bar]))
    gamma = (uav.c * pn.t_s)**2 - d_u**2
    if gamma <= 0 or omega_u / np.tan(uav.theta) > pn.h_max:
        if gamma <= 0:
            print(f"problem infeasible due to too long horizontal distance to fly")
        if omega_u / np.tan(uav.theta) > pn.h_max:
            print(f"problem infeasible due to insufficient height to fly: maximum height = {pn.h_max}, minimum "
                  f"coverage height = {omega_u / np.tan(uav.theta)}")
        return MpRelaxProblem(None, None, None, None, None, None, None, False)
    else:
        gamma_u = np.sqrt(gamma)  # \Gamma(\phi_u) in the paper
        h = min(pn.h_max, max(uav.h_bar - gamma_u, pn.h_min, omega_u / np.tan(uav.theta)))
        uav.h = h

        # Then, generate the problem object
        a, b, p_u, x_u = [], [], [], []
        for s in sc.slices:
            for ue in s.UEs:
                a.append(s.b_width)
                b_i = pn.g_0 * ue.tilde_g / (
                    pn.sigma * (uav.theta**2) * ((ue.loc_x - uav.x) ** 2 + (ue.loc_y - uav.y) ** 2 + h**2) ** (pn.alpha / 2))
                b.append(b_i)
                p_u.append((np.exp(s.r_sla/s.b_width) - 1)/b_i)
                x_u.append(math.ceil(ue.tilde_r))
        a, b, p_u, x_u = np.array(a), np.array(b), np.array(p_u), np.array(x_u)
        c = a
        P = pn.p_max  # this is usually the case.
        B_tot = pn.b_tot
        return MpRelaxProblem(a, b, x_u, c, B_tot, P, p_u, True)


if __name__ == '__main__':
    from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_p_BFGS, optimize_x_cvx

    # np.random.seed(0)
    sc = create_scenario(20, 500)
    sc.pn.b_tot = 400
    # save_scenario(sc, 'sc_2_slices.pickle')
    # sc = load_scenario('sc_2_slices.pickle')
    prob = scenario_to_problem(sc)
    if not prob.is_feasible:
        raise ValueError("The problem is infeasible")
    else:
        lam, rho = 1, 1
        # f, x, p = bcd_al(prob, lam, rho, subx_optimizer=gpm_x, subp_optimizer=optimize_p_BFGS)
        # print(f"GPM-CVX: f = {f}, x = {x}, y = {p}")
        t = time.perf_counter()
        f, x, p, n = Lag_SQP_alg(prob, prob.x_u, prob.p_u)
        print(f"DAL: xp - P = {np.dot(x, p) - prob.P}")
        print(f"AL_SQP finished in {time.perf_counter() - t} sec, f = {f: .8f}")

        t = time.perf_counter()
        # use static allocation as a warm start
        f_static, x_static, p_static = static_power_alloc(prob)
        f, x, p, _, _, _, _ = DAL_alg(prob, x_static, p_static, subx_optimizer=optimize_x_cvx,
                                      subp_optimizer=optimize_p_BFGS)
        print(f"DAL: xp - P = {np.dot(x, p) - prob.P}")
        print(f"DAL finished in {time.perf_counter() - t} sec, f = {f: .8f}, f_static = {f_static}")

    # check optimality






