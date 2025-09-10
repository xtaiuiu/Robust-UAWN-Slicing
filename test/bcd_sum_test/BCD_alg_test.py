import unittest

import numpy as np
from mealpy.math_based import SHIO

from algorithms.MP_Relax_algorithms.MP_Relax_problem import create_optimization_instance
from algorithms.MP_Relax_algorithms.main_algorithm.BCD.BCD_algorithm import bcd_al
from algorithms.MP_Relax_algorithms.main_algorithm.BCD.Gradient_descent_BCD import pgd_BCD
from algorithms.MP_Relax_algorithms.main_algorithm.BCD.Heuristic_BCD import BCD_by_heuristic
from algorithms.MP_Relax_algorithms.main_algorithm.BCD.Solver_BCD import solver_BCD
from scenarios.scenario_creators import create_scenario, scenario_to_problem, save_scenario, load_scenario


class My2BCDTestCase(unittest.TestCase):

    def test_bcd_al_random(self):
        #np.random.seed(0)
        prob = create_optimization_instance(n=200)
        lam, rho = 1, 1
        f, x, p, _, _ = bcd_al(prob, lam, rho, prob.x_u, prob.p_u)
        print(f"f = {f}, x = {x}, y = {p}")
        self.assertLessEqual(np.dot(x, prob.c), prob.B_tot + 1e-4)
        self.assertGreaterEqual(np.min(x - prob.x_u) + 1e-4, 0)
        self.assertGreaterEqual(np.min(p - prob.p_u) + 1e-4, 0)

    def test_bcd_al_real_sc(self):
        np.random.seed(0)
        # sc = create_scenario(2000, 500)
        # save_scenario(sc, 'sc_debug_2.pickle')
        sc = load_scenario('sc_debug_2.pickle')
        prob = scenario_to_problem(sc)
        lam, rho = 1, 1
        f, x, p, _, _ = bcd_al(prob, lam, rho, prob.x_u, prob.p_u)
        print(f"f = {f}, x = {x}, y = {p}")
        self.assertLessEqual(np.dot(x, prob.c), prob.B_tot + 1e-4)
        self.assertGreaterEqual(np.min(x - prob.x_u) + 1e-4, 0)
        self.assertGreaterEqual(np.min(p - prob.p_u) + 1e-4, 0)

        print("__________________________check feasibility of BCD solution______________________________")

        # check feasibility of DAL warm solution
        print(f"c^T x - B_tot = {prob.c @ x - prob.B_tot}")
        print(f"x^p - P = {x @ p - prob.P}")
        print(f"min(x - prob.x_u) = {np.min(x - prob.x_u)}")
        print(f"min(p - prob.p_u) = {np.min(p - prob.p_u)}")

    def test_bcd_pgd_real_sc(self):
        np.random.seed(0)
        sc = create_scenario(2, 500)
        save_scenario(sc, 'sc_debug_2.pickle')
        # sc = load_scenario('sc_debug_BCD.pickle')
        prob = scenario_to_problem(sc)
        lam, rho = 1, 1
        f, x, p = pgd_BCD(prob, lam, rho, prob.x_u, prob.p_u)
        print(f"f = {f}, x = {x}, y = {p}")
        self.assertLessEqual(np.dot(x, prob.c), prob.B_tot + 1e-4)
        self.assertGreaterEqual(np.min(x - prob.x_u) + 1e-4, 0)
        self.assertGreaterEqual(np.min(p - prob.p_u) + 1e-4, 0)

        print("__________________________check feasibility of BCD pgd solution______________________________")

        # check feasibility of DAL warm solution
        print(f"c^T x - B_tot = {prob.c @ x - prob.B_tot}")
        print(f"x^p - P = {x @ p - prob.P}")
        print(f"min(x - prob.x_u) = {np.min(x - prob.x_u)}")
        print(f"min(p - prob.p_u) = {np.min(p - prob.p_u)}")

    def test_bcd_solver_real_sc(self):
        np.random.seed(0)
        # sc = create_scenario(5, 500)
        # save_scenario(sc, 'sc_debug_2.pickle')
        sc = load_scenario('sc_debug_2.pickle')
        prob = scenario_to_problem(sc)
        lam, rho = 1, 1
        f, x, p = solver_BCD(prob, lam, rho, prob.x_u, prob.p_u)
        print(f"f = {f}, x = {x}, y = {p}")
        self.assertLessEqual(np.dot(x, prob.c), prob.B_tot + 1e-4)
        self.assertGreaterEqual(np.min(x - prob.x_u) + 1e-4, 0)
        self.assertGreaterEqual(np.min(p - prob.p_u) + 1e-4, 0)

        print("__________________________check feasibility of BCD solver solution______________________________")

        # check feasibility of DAL warm solution
        print(f"c^T x - B_tot = {prob.c @ x - prob.B_tot}")
        print(f"x^p - P = {x @ p - prob.P}")
        print(f"min(x - prob.x_u) = {np.min(x - prob.x_u)}")
        print(f"min(p - prob.p_u) = {np.min(p - prob.p_u)}")

    def test_bcd_heu_real_sc(self):
        np.random.seed(0)
        # sc = create_scenario(50, 500, p_max=10000, b_tot=10000)
        # save_scenario(sc, 'sc_debug_BCD.pickle')
        sc = load_scenario('sc_debug_2.pickle')
        lam, rho = 1, 1
        prob = scenario_to_problem(sc)
        n = len(prob.a)
        model_shio = SHIO.OriginalSHIO(epoch=1000, pop_size=200)
        f_shio, x_shio = BCD_by_heuristic(prob, lam, rho, model_shio)
        print(f"f_heu = {f_shio: .8f}")
        self.assertLessEqual(np.dot(x_shio[:n], prob.c), prob.B_tot + 1e-4)
        self.assertGreaterEqual(np.min(x_shio[:n] - prob.x_u) + 1e-4, 0)
        self.assertGreaterEqual(np.min(x_shio[n:] - prob.p_u) + 1e-4, 0)

        print("______________________check feasibility of shio solution______________________________")
        # check feasibility of shio solution
        print(f"c^T x - B_tot = {prob.c @ x_shio[:n] - prob.B_tot}")
        print(f"x^p - P = {x_shio[:n] @ x_shio[n:] - prob.P}")
        print(f"min(x - prob.x_u) = {np.min(x_shio[:n] - prob.x_u)}")
        print(f"min(p - prob.p_u) = {np.min(x_shio[n:] - prob.p_u)}")


if __name__ == '__main__':
    unittest.main()
