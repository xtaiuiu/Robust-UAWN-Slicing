import logging
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mealpy import PSO, FloatVar
from mealpy.math_based import SHIO, CircleSA, GBO, HC, INFO, SCA
from mealpy.utils.problem import Problem
from pathlib import Path

from algorithms.MP_Relax_algorithms.benchmark_algorithms.Heuristic_algorithm import optimize_by_heuristic
from algorithms.MP_Relax_algorithms.benchmark_algorithms.Lagrange_SQP import Lag_SQP_alg
from algorithms.MP_Relax_algorithms.benchmark_algorithms.SCA import sca_majorant_with_backtracking
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.Benchmark_static_power import static_power_alloc
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import DAL_alg
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_x_cvx, optimize_p_BFGS
from scenarios.scenario_creators import create_scenario, scenario_to_problem

LOG_LEVEL = logging.INFO  # Change to DEBUG for more verbose output
BASE_DIR = Path(__file__).resolve().parent

def run_save(n_repeats=100):
    logging.disable(logging.INFO)
    t = time.perf_counter()
    print(f"****************************simulation started at {time.asctime()} *******************************")
    step = 10
    powers = np.arange(2, 9) * step
    pd_columns = ['DAL_P', 'SHIO', 'FPS', 'AIW-PSO', 'AL-SQP']
    df_power_UE_avg = pd.DataFrame(np.zeros((len(powers), len(pd_columns))), columns=pd_columns)

    for repeat in range(n_repeats):
        print(f" ###################################### repeat = {repeat} #######################################")
        df_rate = pd.DataFrame(np.zeros((len(powers), len(pd_columns))), columns=pd_columns)
        for i in range(len(powers)):
            print(f" ####################### repeat = {repeat}, power = {powers[i]} #############################")
            sc = create_scenario(50, 100, b_tot=100, p_max=powers[i])
            prob = scenario_to_problem(sc)

            f_static, x_static, p_static = static_power_alloc(prob)

            model_pso = GBO.OriginalGBO(epoch=400, pop_size=50)
            f_pso, x_pso = optimize_by_heuristic(prob, model_pso)

            model_shio = SHIO.OriginalSHIO(epoch=400, pop_size=50)
            f_shio, x_shio = optimize_by_heuristic(prob, model_shio)
            f_sqp, x_sqp, p_sqp, _ = Lag_SQP_alg(prob, x_static, p_static)
            # f_sca, x_sca, p_sca, _ = sca_majorant_with_backtracking(prob, x0=x_static, p0=p_static, tol=1e-6,
            #                                                         verbose=False)
            # l = len(prob.a)
            # if f_static < min(f_pso, f_shio):
            #     x_init, p_init = x_static, p_static
            # elif f_pso < min(f_static, f_shio):
            #     x_init, p_init = x_pso[:l], x_pso[l:]
            # else:
            #     x_init, p_init = x_shio[:l], x_shio[l:]

            f_warm, x_warm, p_warm, _, _, _, _ = DAL_alg(prob, x_static, p_static, subx_optimizer=optimize_x_cvx,
                                                         subp_optimizer=optimize_p_BFGS)
            df_rate.iloc[i, 0] = f_warm
            df_rate.iloc[i, 1] = f_shio
            df_rate.iloc[i, 2] = f_static
            df_rate.iloc[i, 3] = f_pso
            df_rate.iloc[i, 4] = f_sqp
            # df_rate.iloc[i, 5] = f_sca
            print(f"f_DAL = {f_warm: .8f}, f_shio = {f_shio: .8f}, f_fps = {f_static: .8f}, f_pso = {f_pso: .8f}, "
                  f"f_sqp = {f_sqp: .8f}")

        df_power_UE_avg += df_rate
    df_power_UE_avg /= (-n_repeats)
    excel_path = BASE_DIR / "df_power_UE_avg_major_100.xlsx"
    df_power_UE_avg.to_excel(excel_path)
    print(f"compare P finished in {time.perf_counter() - t}")
    df_power_UE_avg.plot()
    plt.show()

def load_plot():
    fontsize = 18
    step = 10
    ##################### df_DAL_iter_avg #######################
    excel_path = BASE_DIR / "df_power_UE_avg_major_100.xlsx"
    df_DAL_iter_avg = pd.read_excel(excel_path, index_col=0)
    df_DAL_iter_avg /= 50
    df_DAL_iter_avg.columns = ['RUNs', 'SHIO', 'FPS', 'GBO', 'SQP', 'SCA']
    df_DAL_iter_avg = df_DAL_iter_avg[['RUNs', 'SHIO', 'GBO', 'SQP', 'SCA']]
    # df_DAL_iter_avg = df_DAL_iter_avg[['RUNs', 'SHIO', 'GBO']]  # conference version
    G = np.arange(df_DAL_iter_avg.shape[0])
    plt.rcParams.update({'font.size': fontsize})

    ax = df_DAL_iter_avg.plot(legend=True, lw=2, xlabel=r'Maximum antenna power $P^{t,\max}$ (W)', ylabel='Avg. data rate (Mbps)',
                              fontsize=fontsize, style=["r-s", "m-.d", "c-->", "k:*", 'b-o'], markersize=10, grid=True,
                              markerfacecolor='none')
    ax.set_xticks(G)
    # plt.ylim((0.295, 0.5))

    ax.set_xticklabels([str((i + 2) * step) for i in G], fontsize=fontsize)
    ax.legend(bbox_to_anchor=(0.8, 0.36), loc='center')

    plt.show()


if __name__ == '__main__':
    # run_save(100)
    load_plot()
