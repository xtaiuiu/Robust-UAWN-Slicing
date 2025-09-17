# This file compares the runtime of four different algorithms:
# 1. RUNs
# 2. SCA
# 3. SHIO/GBO
# 4. AL-SQP
import logging
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mealpy import PSO, FloatVar
from mealpy.math_based import SHIO, GBO
from mealpy.utils.problem import Problem

from algorithms.MP_Relax_algorithms.benchmark_algorithms.Heuristic_algorithm import optimize_by_heuristic
from algorithms.MP_Relax_algorithms.benchmark_algorithms.Lagrange_SQP import Lag_SQP_alg
from algorithms.MP_Relax_algorithms.benchmark_algorithms.SCA import sca_majorant_with_backtracking
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.Benchmark_static_power import static_power_alloc
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import DAL_alg
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_x_cvx, optimize_p_BFGS
from scenarios.scenario_creators import create_scenario, scenario_to_problem

LOG_LEVEL = logging.INFO  # Change to DEBUG for more verbose output


def run_save(n_repeats=100):
    logging.disable(logging.INFO)
    t = time.perf_counter()
    print(f"****************************simulation started at {time.asctime()} *******************************")
    n_UEs = np.array([2, 4, 6, 8, 10])
    # n_UEs = np.array([12])
    pd_columns = ['AL-SQP', 'RUNs']
    df_rate_avg = pd.DataFrame(np.zeros((len(n_UEs), len(pd_columns))), columns=pd_columns)
    df_time_avg = pd.DataFrame(np.zeros((len(n_UEs), len(pd_columns))), columns=pd_columns)

    for repeat in range(n_repeats):
        print(f" ###################################### repeat = {repeat} #######################################")
        df_rate = pd.DataFrame(np.zeros((len(n_UEs), len(pd_columns))), columns=pd_columns)
        df_time = pd.DataFrame(np.zeros((len(n_UEs), len(pd_columns))), columns=pd_columns)
        for i in range(len(n_UEs)):
            print(f" ####################### repeat = {repeat}, nUE = {n_UEs[i]} #############################")
            sc = create_scenario(n_UEs[i], 100, b_tot=100, p_max=50)
            prob = scenario_to_problem(sc)
            f_static, x_static, p_static = static_power_alloc(prob)

            t = time.perf_counter()
            f_sqp, x_sqp, p_sqp, _ = Lag_SQP_alg(prob, x_static, p_static)
            t_sqp = time.perf_counter() - t
            df_time.iloc[i, 0] = t_sqp
            df_rate.iloc[i, 0] = f_sqp

            # t = time.perf_counter()
            # f_sca, x_sca, p_sca, _ = sca_majorant_with_backtracking(prob, x0=x_static, p0=p_static, tol=1e-6,
            #                                                         verbose=False)
            # df_time.iloc[i, 1] = time.perf_counter() - t
            # df_rate.iloc[i, 1] = f_sca

            t = time.perf_counter()
            f_runs, x_runs, p_runs, _, _, _, _ = DAL_alg(prob, x_static, p_static, subx_optimizer=optimize_x_cvx,
                                                         subp_optimizer=optimize_p_BFGS)
            t_runs = time.perf_counter() - t
            df_time.iloc[i, 1] = t_runs
            df_rate.iloc[i, 1] = f_runs

            print(f"f_RUNs = {f_runs: .8f}, f_fps = {f_static: .8f}, "
                  f"f_sqp = {f_sqp: .8f}")
            print(f"t_RUNs = {t_runs: .8f}, t_sqp = {t_sqp: .8f}")

        df_rate_avg += df_rate
        df_time_avg += df_time
    df_rate_avg /= (-n_repeats)
    df_time_avg /= n_repeats
    df_rate_avg.to_excel("df_rate_avg_major_100.xlsx")
    df_time_avg.to_excel("df_time_avg_major_100.xlsx")
    print(f"finished in {time.perf_counter() - t}")
    df_rate_avg.plot()
    df_time_avg.plot()
    plt.show()


if __name__ == "__main__":
    run_save(100)
