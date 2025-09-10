import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import DAL_alg, optimize_x_cvx
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.GPM_algorithm import gpm_x
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_p_BFGS, optimize_p_cvx
from scenarios.scenario_creators import scenario_to_problem, load_scenario, create_scenario, save_scenario


def run_save(n_repeats=100):
    t = time.perf_counter()
    print(f"****************************simulation started at {time.asctime()} *******************************")
    """
    Compute the number of the outer iterations of DAL algorithm
    :return:
    """
    step = 500
    eps_arr = np.array([1e-4, 1e-8, 1e-12])
    n_UEs = np.arange(1, 7) * step
    df_DAL_iter_avg = pd.DataFrame(np.zeros((len(n_UEs), len(eps_arr))), columns=eps_arr)
    df_DAL_time_avg = pd.DataFrame(np.zeros((len(n_UEs), len(eps_arr))), columns=eps_arr)
    df_BCD_iter_avg = pd.DataFrame(np.zeros((len(n_UEs), len(eps_arr))), columns=eps_arr)
    df_BCD_time_avg = pd.DataFrame(np.zeros((len(n_UEs), len(eps_arr))), columns=eps_arr)

    for repeat in range(n_repeats):
        print(f" ####################### repeat = {repeat} ##############################")
        df_DAL_iter = pd.DataFrame(np.zeros((len(n_UEs), len(eps_arr))),columns=eps_arr)
        df_DAL_time = pd.DataFrame(np.zeros((len(n_UEs), len(eps_arr))), columns=eps_arr)
        df_BCD_iter = pd.DataFrame(np.zeros((len(n_UEs), len(eps_arr))), columns=eps_arr)
        df_BCD_time = pd.DataFrame(np.zeros((len(n_UEs), len(eps_arr))), columns=eps_arr)
        for i in range(len(n_UEs)):
            print(f" ####################### repeat = {repeat}, n_ue = {n_UEs[i]} ##############################")
            sc = create_scenario(n_UEs[i], 500, b_tot=1000, p_max=1000)
            prob = scenario_to_problem(sc)
            # check problem feasibility:

            for j in range(len(eps_arr)):
                print(f" ####################### repeat = {repeat}, n_ue = {n_UEs[i]}, eps = {eps_arr[j]} ##############################")
                t1 = time.perf_counter()
                f, x, p, n, n_inner_x, n_inner_p, bcd_times = DAL_alg(prob, prob.x_u, prob.p_u,
                                                                      subx_optimizer=optimize_x_cvx,
                                                                      subp_optimizer=optimize_p_BFGS, eps=eps_arr[j])
                df_DAL_iter.iloc[i, j] = n
                df_DAL_time.iloc[i, j] = time.perf_counter() - t1
                df_BCD_iter.iloc[i, j] = np.sum(n_inner_x + n_inner_p)
                df_BCD_time.iloc[i, j] = np.sum(bcd_times)
        df_DAL_iter_avg += df_DAL_iter
        df_DAL_time_avg += df_DAL_time
        df_BCD_iter_avg += df_BCD_iter
        df_BCD_time_avg += df_BCD_time
    df_DAL_iter_avg /= n_repeats
    df_DAL_time_avg /= n_repeats
    df_BCD_iter_avg /= n_repeats
    df_BCD_time_avg /= n_repeats

    df_DAL_iter_avg.to_excel("df_DAL_iter_avg_3_3.xlsx")
    df_DAL_time_avg.to_excel("df_DAL_time_avg_3_3.xlsx")
    df_BCD_iter_avg.to_excel("df_BCD_iter_avg_3_3.xlsx")
    df_BCD_time_avg.to_excel("df_BCD_time_avg_3_3.xlsx")

    df_DAL_iter_avg.plot()
    df_DAL_time_avg.plot()
    df_BCD_iter_avg.plot()
    df_BCD_time_avg.plot()
    print(f"*************************** simulation finished in {time.perf_counter() - t} seconds ***********************")

    plt.show()

def load_plot():
    fontsize = 20
    step = 500
    ##################### df_DAL_iter_avg #######################
    df_DAL_iter_avg = pd.read_excel('df_DAL_iter_avg.xlsx', index_col=0)
    df_DAL_iter_avg.columns = [r'$\epsilon=10^{-4}$', r'$\epsilon=10^{-8}$', r'$\epsilon=10^{-12}$']
    G = np.arange(df_DAL_iter_avg.shape[0])
    plt.rcParams.update({'font.size': fontsize})

    ax = df_DAL_iter_avg.plot(legend=True, lw=2, xlabel='Number of UEs', ylabel='No. of outer iterations of DAL',
                         fontsize=fontsize, style=["r-s", "m-.d", "c-->"], markersize=10, grid=True,
                         markerfacecolor='none')
    ax.set_xticks(G)
    # plt.ylim((0.295, 0.5))
    ax.set_xticklabels([str((i + 1) * step) for i in G], fontsize=fontsize)

    ##################### df_DAL_time_avg #######################
    df_DAL_time_avg = pd.read_excel('df_DAL_time_avg.xlsx', index_col=0)
    df_DAL_time_avg.columns = [r'$\epsilon=10^{-4}$', r'$\epsilon=10^{-8}$', r'$\epsilon=10^{-12}$']
    G = np.arange(df_DAL_time_avg.shape[0])
    plt.rcParams.update({'font.size': fontsize})

    ax = df_DAL_time_avg.plot(legend=True, lw=2, xlabel='Number of UEs', ylabel='Convergence time of DAL (s)',
                              fontsize=fontsize, style=["r-s", "m-.d", "c-->"], markersize=10, grid=True,
                              markerfacecolor='none')
    ax.set_xticks(G)
    # plt.ylim((0.295, 0.5))
    ax.set_xticklabels([str((i + 1) * step) for i in G], fontsize=fontsize)

    ##################### df_BCD_iter_avg #######################
    df_BCD_iter_avg = pd.read_excel('df_BCD_iter_avg.xlsx', index_col=0)
    df_BCD_iter_avg.columns = [r'$\epsilon=10^{-4}$', r'$\epsilon=10^{-8}$', r'$\epsilon=10^{-12}$']
    G = np.arange(df_BCD_iter_avg.shape[0])
    plt.rcParams.update({'font.size': fontsize})

    ax = df_BCD_iter_avg.plot(legend=True, lw=2, xlabel='Number of UEs', ylabel='No. of inner iterations of DAL',
                              fontsize=fontsize, style=["r-s", "m-.d", "c-->"], markersize=10, grid=True,
                              markerfacecolor='none')
    ax.set_xticks(G)
    # plt.ylim((0.295, 0.5))
    ax.set_xticklabels([str((i + 1) * step) for i in G], fontsize=fontsize)

    ##################### df_BCD_time_avg #######################
    # df_BCD_time_avg = pd.read_excel('df_BCD_time_avg.xlsx', index_col=0)
    # df_BCD_time_avg.columns = [r'$\epsilon=1e-4$', r'$\epsilon=1e-8$', r'$\epsilon=1e-12$']
    # G = np.arange(df_BCD_time_avg.shape[0])
    # plt.rcParams.update({'font.size': fontsize})
    #
    # ax = df_BCD_time_avg.plot(legend=True, lw=2, xlabel='Number of UEs', ylabel='Time of inner iterations',
    #                           fontsize=fontsize, style=["r-s", "m-.d", "c-->"], markersize=10, grid=True,
    #                           markerfacecolor='none')
    # ax.set_xticks(G)
    # # plt.ylim((0.295, 0.5))
    # ax.set_xticklabels([str((i + 1) * step) for i in G], fontsize=fontsize)


    plt.show()


if __name__ == '__main__':
    # run_save(1)
    load_plot()
    # run the run_save function, wherein the number of repeats is set to 100
