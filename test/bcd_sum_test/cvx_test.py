import time
import unittest
import numpy as np
from mealpy.math_based import SHIO

from algorithms.MP_Relax_algorithms.benchmark_algorithms.Heuristic_algorithm import optimize_by_heuristic
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import create_optimization_instance, optimize_x_cvx, optimize_p_cvx, \
    DAL_alg, optimize_x_gradient_descent, OptimizationProblem, create_fixed_optmization_problem


class MyTestCase(unittest.TestCase):
    def test_optimize_x_cvx_random(self):
        prob = create_optimization_instance()
        p = prob.p_u
        lam, rho = 1.0, 1.0
        f_val, var = optimize_x_cvx(prob, lam, rho, p)
        print(f"test_x_cvx_random: f_val = {f_val}, var = {var}")

        # test feasibility of the solution
        self.assertAlmostEqual(np.dot(prob.c, var), prob.B_tot, delta=1e-4)  # c^x == B_tot
        self.assertTrue(np.all(var >= prob.x_u))  # x >= x_u

        # test the optimality of the solution
        x0 = prob.x_u
        x0[-1] = (prob.B_tot - np.dot(prob.c[:-1], var[:-1]))/prob.c[-1]
        self.assertGreaterEqual(prob.Lagrangian_value(x0, p, lam, rho), f_val)

    def test_optimize_x_cvx_fixed(self):
        prob = create_fixed_optmization_problem()
        p = prob.p_u
        lam, rho = 1.0, 1.0
        f_val, var = optimize_x_cvx(prob, lam, rho, p)
        print(f"test_x_cvx_fixed: f_val = {f_val}, var = {var}")

        # test feasibility of the solution
        self.assertAlmostEqual(np.dot(prob.c, var), prob.B_tot, delta=1e-4)  # c^x == B_tot
        self.assertTrue(np.all(var >= prob.x_u))  # x >= x_u

        # test the optimality of the solution
        x0 = prob.x_u
        x0[-1] = (prob.B_tot - np.dot(prob.c[:-1], var[:-1])) / prob.c[-1]
        self.assertGreaterEqual(prob.Lagrangian_value(x0, p, lam, rho), f_val)

    def test_optimize_p_cvx_random(self):
        prob = create_optimization_instance()
        x = prob.x_u
        lam, rho = 1.0, 1.0
        f_val, var = optimize_p_cvx(prob, lam, rho, x)
        print(f"test_cvx_p_random: f_val = {f_val}, var = {var}")

        # test feasibility of the solution
        self.assertTrue(np.all(var >= prob.p_u))  # p >= p_u

        # test the optimality of the solution
        self.assertGreaterEqual(prob.Lagrangian_value(x, prob.p_u, lam, rho), f_val)

    def test_optimize_p_cvx_fixed(self):
        prob = create_fixed_optmization_problem()
        x = prob.x_u
        lam, rho = 1.0, 1.0
        f_val, var = optimize_p_cvx(prob, lam, rho, x)
        print(f"test_cvx_p_fixed: f_val = {f_val}, var = {var}")

        # test feasibility of the solution
        self.assertTrue(np.all(var >= prob.p_u))  # p >= p_u

        # test the optimality of the solution
        self.assertGreaterEqual(prob.Lagrangian_value(x, prob.p_u, lam, rho), f_val)


    def test_optimize_x_gradient_descent(self):
        np.random.seed(0)
        prob = create_optimization_instance(n=10000)
        y = prob.p_u
        lam, rho = 1.0, 1.0

        t = time.perf_counter()
        f_val, var = optimize_x_gradient_descent(prob, lam, rho, y, dynamic_rule=True)
        print(f"GD finished in {time.perf_counter() - t} seconds.")
        print(f"f_val_GD = {f_val}, var = {0}")

        t = time.perf_counter()
        f_val, var = optimize_x_cvx(prob, lam, rho, y)
        print(f"CVX finished in {time.perf_counter() - t} seconds.")
        print(f"f_val_CVX = {f_val}, var = {0}")


    def test_optimize_by_heuristic(self):
        f_pdd, f_heu = 0, 0
        for i in range(1):

            k = np.random.randint(1, 3)
            prob = create_optimization_instance(k)
            model_shio = SHIO.OriginalSHIO(epoch=20, pop_size=200)
            f, var = optimize_by_heuristic(prob, model_shio)
            n = len(prob.a)
            x, y = var[:n], var[n:]
            print(f"f = {f}")
            self.assertLessEqual(np.dot(x, prob.c), prob.B_tot)
            self.assertLessEqual(np.dot(x, y), prob.P)
            self.assertTrue(all(a >= b for a, b in zip(x, prob.x_u)))
            self.assertTrue(all(a >= b for a, b in zip(y, prob.p_u)))

    def test_optimize_1dim_prob(self):
        prob = OptimizationProblem([1, 2], [2, 1], [0.1, 0.1], [0.1, 0.1], 10, 10, [0.1, 0.1], )
        model_shio = SHIO.OriginalSHIO(epoch=4000, pop_size=400)
        f, var = optimize_by_heuristic(prob, model_shio)
        print(f" f = {f}, var = {var}")
        f, x, y = DAL_alg(prob, prob.x_u, prob.p_u)
        print(f"f = {f}, x = {x}, y = {y}")


if __name__ == '__main__':
    unittest.main()
