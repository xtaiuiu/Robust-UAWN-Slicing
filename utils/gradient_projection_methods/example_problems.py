import numpy as np
# bound constrained convex problems:
# \min\limits_{x} 1/2 * (a^T x - p)^2 + \lamda *(a^T x - p) - \sum\limits_{i=1}^n w_n ln (1+b_n x_n)
# s.t. x \ge q
class BoundProblem():
    def __init__(self, a, b, lam, p, q, w):
        self.a = a
        self.b = b
        self.lam = lam
        self.p = p
        self.q = q
        self.w = w

    def f(self, x):
        """
        Evaluate f(x) at the x
        :param x:
        :return:
        """
        z = np.dot(self.a, x) - self.p
        res = 0.5 * z**2 + self.lam * z
        for i in range(len(x)):
            res -= self.w[i] * np.log1p(self.b[i] * x[i])
        return res

    def g(self, x):
        # gradient of the function at x
        z = np.dot(self.a, x) - self.p
        s = np.array([self.b[i]*self.w[i]/(1 + self.b[i] * x[i]) for i in range(len(x))])
        return z*self.a + self.lam * self.a - s


# simplex constrained problem:
# \min\limits_{x} 0.5 * (a^T x - p)^2 + b^T x
# s.t. x >= c,
#      d^T x = e
class SimplexProblem():

    def __init__(self, a, b, c, d, e, p):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.p = p

    def f(self, x):
        return 0.5 * (np.dot(self.a, x) - self.p)**2 + np.dot(self.b, x)

    def g(self, x):
        return (np.dot(self.a, x) - self.p)*self.a + self.b