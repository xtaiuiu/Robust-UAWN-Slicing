import unittest

import numpy as np

from utils.projction import proj_nonnegative_half_space_lb, proj_nonnegative_half_space_QP


class MyTestCase(unittest.TestCase):
    def test_proj_nonnegative_half_space(self):
        for n in np.arange(2, 101)*10:
            a = np.random.uniform(0.1, 4, n)
            y = np.random.uniform(1, 10, n)
            r = a@y + np.random.uniform(10, 20)
            l = np.random.uniform(0.1, 0.9)*np.ones(n)*np.min(y)
            y = np.random.uniform(-10, 10, n)

            x = proj_nonnegative_half_space_lb(a, y, r, l)
            x_q = proj_nonnegative_half_space_QP(a, y, r, l)
            print(f" n = {n}, norm = {np.linalg.norm(x - x_q)}")
            self.assertLessEqual(a@x, r + 1e-8)
            self.assertAlmostEqual(0, np.linalg.norm(x - x_q, ord=np.inf), delta=1e-1)


if __name__ == '__main__':
    unittest.main()
