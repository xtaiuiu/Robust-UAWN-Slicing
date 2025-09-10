import logging
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mealpy import PSO, FloatVar
from mealpy.evolutionary_based import DE
from mealpy.math_based import SHIO, AOA, CEM, CGO, PSS, RUN, GBO
from mealpy.physics_based import ArchOA
from mealpy.utils.problem import Problem

from algorithms.MP_Relax_algorithms.benchmark_algorithms.Heuristic_algorithm import optimize_by_heuristic
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.Benchmark_static_power import static_power_alloc
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import DAL_alg
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_x_cvx, optimize_p_BFGS
from scenarios.scenario_creators import create_scenario, scenario_to_problem


def run_save(n_repeats=100):
    logging.disable(logging.INFO)
    t = time.perf_counter()
    eps_arr = np.array([0.01, 0.05, 0.1])
    step = 0.01
    delta_arr = np.arange(0, 7) * step
    sc = create_scenario(50, 500, b_tot=100, p_max=50)
    print(f"****************************simulation started at {time.asctime()} *******************************")
    df_robust_avg = pd.DataFrame(np.zeros((len(delta_arr), len(eps_arr))), columns=eps_arr)

    for repeat in range(n_repeats):
        print(f" ###################################### repeat = {repeat} #######################################")
        df_rate = pd.DataFrame(np.zeros((len(delta_arr), len(eps_arr))), columns=eps_arr)
        for i in range(len(delta_arr)):
            print(f" ####################### repeat = {repeat}, delta = {delta_arr[i]} #############################")
            for j in range(len(eps_arr)):
                sc.set_UE_Delta_g(eps_arr[j], delta_arr[i])
                prob = scenario_to_problem(sc)
                f_opt, x_opt, p_opt = static_power_alloc(prob)
                x_init, p_init = x_opt, p_opt
                f_warm, _, _, _, _, _, _ = DAL_alg(prob, x_init, p_init, subx_optimizer=optimize_x_cvx,
                                                          subp_optimizer=optimize_p_BFGS, eps=1e-8)
                df_rate.iloc[i, j] = f_warm
                sc.reset_UE_Delta_g()
        df_robust_avg += df_rate
    df_robust_avg /= (-n_repeats)
    df_robust_avg.to_excel("df_robust_avg.xlsx")
    print(f"compare B finished in {time.perf_counter() - t}")
    df_robust_avg.plot()
    plt.show()


def load_plot():
    fontsize = 20
    step = 0.01
    ##################### df_robust_avg #######################
    df_DAL_iter_avg = pd.read_excel('df_PoR_avg.xlsx', index_col=0)
    df_DAL_iter_avg.columns = np.array([r"$\vartheta = 0.01$", r"$\vartheta = 0.05$", r"$\vartheta=0.1$"])
    G = np.arange(df_DAL_iter_avg.shape[0])
    plt.rcParams.update({'font.size': fontsize})

    ax = df_DAL_iter_avg.plot(legend=True, lw=2, xlabel=r'Channel estimation error $\delta_{ij}$', ylabel='PoR (Mbps)',
                              fontsize=fontsize, style=["k-.*", "c--D", "b:^"], markersize=10, grid=True,
                              markerfacecolor='none')
    ax.set_xticks(G)
    # plt.ylim((0.295, 0.5))
    ax.set_xticklabels([str((i) * step) for i in G], fontsize=fontsize)

    ##################### df_robust_price_avg #######################
    df_DAL_iter_avg = pd.read_excel('df_robust_price_avg.xlsx', index_col=0) * 100
    df_DAL_iter_avg.columns = np.array([r"$\vartheta = 0.01$", r"$\vartheta = 0.05$", r"$\vartheta=0.1$"])
    G = np.arange(df_DAL_iter_avg.shape[0])
    plt.rcParams.update({'font.size': fontsize})

    ax = df_DAL_iter_avg.plot(legend=True, lw=2, xlabel=r'Channel estimation error $\delta_{ij}$',
                              ylabel='RPoR (%)',
                              fontsize=fontsize, style=["k-.*", "c--D", "b:^"], markersize=10, grid=True,
                              markerfacecolor='none')
    ax.set_xticks(G)
    # plt.ylim((0.295, 0.5))
    ax.set_xticklabels([str((i) * step) for i in G], fontsize=fontsize)

    plt.show()




if __name__ == '__main__':
    # run_save(100)
    load_plot()
