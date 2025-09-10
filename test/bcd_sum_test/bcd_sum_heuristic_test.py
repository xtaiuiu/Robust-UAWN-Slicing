import unittest

from algorithms.MP_Relax_algorithms.benchmark_algorithms.Heuristic_algorithm import prob_up
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import OptimizationProblem


class MyTestCase(unittest.TestCase):
    def test_prob_ub(self):
        prob = OptimizationProblem([1, 2], [2, 1], [0.1, 0.1], [0.1, 0.1], 10, 10, [0.1, 0.1], )
        ub = prob_up(prob)
        prob = OptimizationProblem(1, 1, 1, 1, 1, 1, 1)
        ub = prob_up(prob)
        print(ub)


if __name__ == '__main__':
    unittest.main()
