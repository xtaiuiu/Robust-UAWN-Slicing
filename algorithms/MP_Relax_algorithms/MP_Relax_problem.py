# Problem class corresponding to the problem MP-Relax.
# \max\limits_{x, p} \sum\limits_{i=1}^n a_i * x_i *log(1 + b_i * p_i),
# s.t. x >= x_u,
# c^T * x = B_tot,
# p >= p_u,
# x^T p = P
import numpy as np


class MpRelaxProblem:  # the class that represents the MP-Relax problem
    def __init__(self, a, b, x_u, c, B_tot, P, p_u, is_feasible):
        """
        初始化优化问题。

        :param a: 系数 a_i 的数组。
        :param b: 系数 b_i 的数组。
        :param x_u: x 的下界数组。
        :param c: 约束 c^T * x = B_tot 中的系数数组。
        :param B_tot: 约束 c^T * x = B_tot 中的常数。
        :param P: 约束 x^T * p = P 中的常数。
        :param p_u: p 的下界数组。
        :param is_feasible: a label that represents whether MP-Relax is feasible.
        If not, don't solve it and set the objective value to -1e8
        """
        self.a = np.atleast_1d(a)
        self.b = np.atleast_1d(b)
        self.x_u = np.atleast_1d(x_u)
        self.c = np.atleast_1d(c)
        self.B_tot = B_tot
        self.P = P
        self.p_u = np.atleast_1d(p_u)
        self.is_feasible = is_feasible


    def objective_function(self, x, p):
        """
        计算目标函数的值。
        :return: 目标函数的值。
        """
        if np.any(p < 0):
            # 抛出一个异常
            raise ValueError("Array contains negative values")
        return -np.sum([self.a[i]*x[i]*np.log1p(self.b[i]*p[i]) for i in range(len(x))])

    # The function value of L(x, p, lam, rho)
    def Lagrangian_value(self, x, p, lam, rho):
        xp_dot = np.dot(x, p)
        return self.objective_function(x, p) + lam*(xp_dot - self.P) + ((xp_dot - self.P) ** 2)/(2 * rho)

    # The gradient of Sub_x at point (x, p). Here, p is a parameter
    def g_x(self, x, p, lam, rho):
        t = lam * p + ((np.dot(x, p) - self.P) / rho) * p
        return np.array([-self.a[i] * np.log1p(self.b[i] * p[i]) + t[i] for i in range(len(x))])

    # The gradient of Sub_p at point(x, p), where x is a parameter
    def g_p(self, x, p, lam, rho):
        t = lam * x + ((np.dot(x, p) - self.P) / rho) * x
        return np.array([-self.a[i] * x[i] * self.b[i] / (1 + self.b[i] * p[i]) + t[i] for i in range(len(x))])


def create_optimization_instance(n=10):
    """
    创建优化问题的实例。
    :param a: 系数 a_i 的数组。
    :param b: 系数 b_i 的数组。
    :param c: x 的下界数组。
    :param d: 约束 d^T * x = e 中的系数数组。
    :param e: 约束 d^T * x = e 中的常数。
    :param p: 约束 1^T * y = p 中的常数。
    :param q: y 的下界数组。
    :return: OptimizationProblem 类的实例。
    """
    x = np.random.randint(low=1, high=20, size=n)
    p = 1 + np.random.random(size=n) * 10

    a = 1 + np.random.random(size=n) * 10
    b = 1 + np.random.random(size=n) * 10

    c = 1 + np.random.random(size=n) * 10
    B_tot = np.dot(c, x)
    P = np.dot(x, p)

    x_min, p_min = np.min(x), np.min(p)
    x_u = np.random.uniform(0.2, 0.8, size=n) * x_min
    p_u = np.random.uniform(0.2, 0.8, size=n) * p_min

    return MpRelaxProblem(a, b, x_u, c, B_tot, P, p_u, True)


def create_fixed_optmization_problem(n=10):
    """
    Create a fixed optimization problem, for test only
    This problem has a feasible solution: x, p = 2*np.ones(n), 2*np.ones(n)
    :return:
    """
    a, b = np.ones(n), np.ones(n)

    c, x_u, p_u = np.ones(n), np.ones(n), np.ones(n)

    x, p = 2*np.ones(n), 2*np.ones(n)

    B_tot, P = np.dot(c, x), np.dot(x, p)

    return MpRelaxProblem(a, b, x_u, c, B_tot, P, p_u)
