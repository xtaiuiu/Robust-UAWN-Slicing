# solve MP-Relax by static power allocation
import time

import numpy as np
from mealpy.math_based import SHIO
from scipy.optimize import linprog



def static_power_alloc(prob, max_iter=100):
    f_opt, x_opt, p_opt = 1e8, prob.x_u, prob.p_u
    p_increment = (prob.P - prob.x_u@prob.p_u)/len(prob.a)/(max_iter*4)
    n, p = 0, prob.p_u
    while n < max_iter:

        p = prob.p_u + n*p_increment * np.ones_like(prob.p_u)  # power allocation
        nu = np.array([-prob.a[i] * np.log1p(prob.b[i] * p[i]) for i in range(len(prob.a))])
        result = linprog(nu, A_ub=np.array([prob.c, p]), b_ub=np.array([prob.B_tot, prob.P]),
                         bounds=[(x_ui, None) for x_ui in prob.x_u])
        if result.success:
            #print(f"f_current = {result.fun}, f_best = {f_opt}")
            if result.fun < f_opt:
                f_opt, x_opt, p_opt = result.fun, result.x, p
        else:
            print("infeasible:　", result.message)
            print(f"prob.c@prob.x_u - prob.B_tot = {prob.c@prob.x_u - prob.B_tot: .8f}, p@prob.x_u - prob.P = {p@prob.x_u - prob.P: .8f}")
            break
        n += 1
    return f_opt, x_opt, p_opt


if __name__ == '__main__':
    from scenarios.scenario_creators import create_scenario, save_scenario, scenario_to_problem, load_scenario
    from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import DAL_alg
    from algorithms.MP_Relax_algorithms.benchmark_algorithms.Heuristic_algorithm import optimize_by_heuristic

    from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_x_cvx, optimize_p_BFGS

    # sc = create_scenario(50, 500, p_max=10000, b_tot=10000)
    # save_scenario(sc, 'sc_debug_BCD.pickle')
    sc = load_scenario('sc_debug_BCD.pickle')
    prob = scenario_to_problem(sc)
    t = time.perf_counter()
    f_opt, x_opt, p_opt = static_power_alloc(prob)
    print(f"static finished in t = {time.perf_counter() - t}")

    print("_________________________check feasibility of static solution___________________________")

    # check feasibility of DAL rand solution
    print(f"c^T x - B_tot = {prob.c @ x_opt - prob.B_tot}")
    print(f"x^p - P = {x_opt @ p_opt - prob.P}")
    print(f"min(x - prob.x_u) = {np.min(x_opt - prob.x_u)}")
    print(f"min(p - prob.p_u) = {np.min(x_opt - prob.p_u)}")

    l = len(prob.a)

    model_shio = SHIO.OriginalSHIO(epoch=2000, pop_size=400)
    # model_shio = PSO.P_PSO(epoch=1000, pop_size=200)
    t1 = time.perf_counter()
    f_shio, x_shio = optimize_by_heuristic(prob, model_shio)
    t2 = time.perf_counter()
    f_warm, x_warm, p_warm, _, _, _ = DAL_alg(prob, x_shio[:l], x_shio[l:], subx_optimizer=optimize_x_cvx,
                                              subp_optimizer=optimize_p_BFGS)
    print("__________________________check feasibility of DAL warm solution______________________________")

    # check feasibility of DAL warm solution
    print(f"c^T x - B_tot = {prob.c @ x_warm - prob.B_tot}")
    print(f"x^p - P = {x_warm @ p_warm - prob.P}")
    print(f"min(x - prob.x_u) = {np.min(x_warm - prob.x_u)}")
    print(f"min(p - prob.p_u) = {np.min(p_warm - prob.p_u)}")
