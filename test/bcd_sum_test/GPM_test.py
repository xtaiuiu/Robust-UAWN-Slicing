import time
import unittest

import numpy as np
from scipy.optimize import linprog

from algorithms.MP_Relax_algorithms.MP_Relax_problem import create_fixed_optmization_problem, \
    create_optimization_instance
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import optimize_x_cvx, optimize_p_cvx
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.GPM_algorithm import gpm_x, gpm_p, gp_momentum_constant_step, \
    gp_x
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_p_BFGS
from scenarios.scenario_creators import create_scenario, scenario_to_problem, save_scenario, load_scenario
from utils.projction import proj_nonnegative_half_space_lb


class MyTestCase(unittest.TestCase):
    def test_GPM_x_random(self):
        prob = create_optimization_instance(n=800)
        lam, rho = 1, 1
        t = time.perf_counter()
        fx, x = gpm_x(prob, lam, rho, prob.p_u, prob.x_u)
        t1 = time.perf_counter()
        print(f"random: fx = {fx}, time = {t1 - t}")
        f_cvx, x_cvx = optimize_x_cvx(prob, lam, rho, prob.p_u)
        print(f"random: fx_cvx = {f_cvx}, time = {time.perf_counter() - t1}")

        # test feasibility of the solution
        self.assertAlmostEqual(0, np.dot(prob.c, x) - prob.B_tot, delta=1e-4)  # c^x <= B_tot
        self.assertTrue(np.all(x + 1e-4*np.ones_like(x) >= prob.x_u))  # x >= x_u

        # test the optimality of the solution
        self.assertAlmostEqual(abs((fx - f_cvx)/fx), 0, delta=1e-4)

    def test_GPM_x_real_scenario(self):
        for i in range(40):
            n = np.random.randint(900, 1000)
            sc = create_scenario(n, 500)
            #save_scenario(sc, "test_GPM_x_scenario.pickle")
            # sc = load_scenario("test_GPM_x_scenario.pickle")
            prob = scenario_to_problem(sc)
            lam, rho = np.random.randint(200, 20000), np.random.uniform(0.0001, 0.001)
            _, p = optimize_p_BFGS(prob, lam, rho, prob.x_u)
            #p = prob.p_u
            nu = np.array([prob.a[i] * np.log1p(prob.b[i] * p[i]) for i in range(len(prob.a))])
            # high-quality of x0
            result = linprog(np.zeros_like(p), A_ub=np.array([prob.c]), b_ub=prob.B_tot, A_eq=np.array([p]), b_eq=prob.P, bounds=[(x_ui, None) for x_ui in prob.x_u])
            x0 = prob.x_u
            if result.success:
                x0 = result.x
            else:
                raise ValueError(f'linprog failed: {result}')
            t = time.perf_counter()
            fx, x = gpm_x(prob, lam, rho, p, x0)
            print(f"g = {np.linalg.norm(prob.g_x(x, p, lam, rho), ord=np.inf)}, nu_min = {np.min(nu/p)}, "
                  f"nu_max = {np.max(nu/p)}, nu_std = {np.var(nu/p)}")
            t1 = time.perf_counter()
            print(f"real_sc: fx = {fx}, time = {t1 - t}")
            f_cvx, x_cvx = optimize_x_cvx(prob, lam, rho, p, x0)
            print(f"real_sc: fx_cvx = {f_cvx}, time = {time.perf_counter() - t1}")

            print(f"g_gpm = {np.linalg.norm(prob.g_x(x, p, lam, rho), ord=np.inf)}, "
                  f"g_cvx = {np.linalg.norm(prob.g_x(x_cvx, p, lam, rho))}")

            # test feasibility of the solution
            self.assertGreaterEqual(1e-8, np.dot(prob.c, x) - prob.B_tot)  # c^x <= B_tot
            self.assertTrue(np.all(x >= prob.x_u))  # x >= x_u

            # test the optimality of the solution
            self.assertAlmostEqual(abs((fx - f_cvx)/fx), 0, delta=1e-2)

    def test_optimize_p_GPM_random(self):
        prob = create_optimization_instance(n=10)
        x = prob.x_u
        lam, rho = 20.0, 0.01
        t1 = time.perf_counter()
        f, p = gpm_p(prob, lam, rho, x)
        t2 = time.perf_counter()
        print(f"test_gpm_p_random: f_val = {f}, time = {t2 - t1}")
        f_val, var = optimize_p_cvx(prob, lam, rho, x)
        print(f"test_cvx_p_random: f_val = {f_val}, time = {time.perf_counter() - t2}")

        # test feasibility of the solution
        self.assertTrue(np.all(p >= prob.p_u))  # p >= p_u

        # test the optimality of the solution
        self.assertAlmostEqual(f, f_val, delta=1e-1)

    def test_optimize_p_GPM_fixed(self):
        prob = create_fixed_optmization_problem(n=10)
        x = prob.x_u
        lam, rho = 20.0, 0.01
        t1 = time.perf_counter()
        f, p = gpm_p(prob, lam, rho, x)
        print(f"test_gpm_p_fixed: f_val = {f}, var = {p}, time = {time.perf_counter() - t1}")
        t2 = time.perf_counter()
        f_val, var = optimize_p_cvx(prob, lam, rho, x)
        print(f"test_cvx_p_fixed: f_val = {f_val}, var = {var}, time={time.perf_counter() - t2}")

        # test feasibility of the solution
        self.assertTrue(np.all(p >= prob.p_u))  # p >= p_u

        # test the optimality of the solution
        self.assertAlmostEqual(f, f_val, delta=1e-1)

    def test_optimize_p_BFGS_random(self):
        prob = create_optimization_instance(n=10)
        x = prob.x_u
        lam, rho = 20.0, 0.01
        t1 = time.perf_counter()
        f, p = optimize_p_BFGS(prob, lam, rho, x)
        t2 = time.perf_counter()
        print(f"test_gpm_p_random: f_val = {f}, time = {t2 - t1}")
        f_val, var = optimize_p_cvx(prob, lam, rho, x)
        print(f"test_cvx_p_random: f_val = {f_val}, time = {time.perf_counter() - t2}")

        # test feasibility of the solution
        self.assertTrue(np.all(p >= prob.p_u))  # p >= p_u

        # test the optimality of the solution
        self.assertAlmostEqual(f, f_val, delta=1e-1)


if __name__ == '__main__':
    unittest.main()
