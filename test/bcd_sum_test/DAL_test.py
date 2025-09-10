import time
import traceback
import unittest

import numpy as np

from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import create_optimization_instance, DAL_alg, \
    create_fixed_optmization_problem, optimize_p_BFGS
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.GPM_algorithm import gpm_x, gpm_p


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


    def test_penalty_bcd_alg_GPM_GPM(self):
        prob = create_fixed_optmization_problem(n=50)
        t1 = time.perf_counter()
        try:
            f, x, p = DAL_alg(prob, subx_optimizer=gpm_x, subp_optimizer=gpm_p)
        except ValueError as e:
            print(f"!!!!!!!!!!!!!!Caught an exception: {e}!!!!!!!!!!!!!")
            # 打印调用栈
            traceback.print_exc()
        print(f"GPM-GPM: f = {f}, time = {time.perf_counter() - t1}")
        self.assertAlmostEqual(np.dot(x, prob.c), prob.B_tot, delta=1e-1)
        self.assertAlmostEqual(np.dot(x, p), prob.P, delta=1e-1)
        self.assertTrue(all(a >= b for a, b in zip(x, prob.x_u)))
        self.assertTrue(all(a >= b for a, b in zip(p, prob.p_u)))


if __name__ == '__main__':
    unittest.main()
