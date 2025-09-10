# optimize sub_x by gradient descent.
# ! this code is not tested.
import numpy as np

from utils.projction import proj_generalized_simplex_lb


def optimize_x_gradient_descent(prob, lam, rho, y, dynamic_rule=False):
    z = np.zeros_like(prob.a)
    a = np.array([prob.a[i]*np.log1p(prob.b[i]*y[i]) for i in range(len(prob.a))])
    n, n_max, eps = 1, 1000, 1e-10
    f = lambda x: np.dot(lam*y-a, x) + ((np.dot(y, x) - prob.P) ** 2) / (2 * rho) - lam * prob.P
    f_old, f_current = 1e8, f(z)
    while abs((f_old - f_current)/f_current) > eps and n < n_max:
        f_old = f_current
        g = (lam + (np.dot(y, z) - prob.P) / rho) * y - a
        if dynamic_rule:
            s_bar, beta, sigma = 2.0, 0.5, 0.8
            m, m_max, x_s = 0, 100, z
            while m < m_max:
                x_s = proj_generalized_simplex_lb(prob.c, z - s_bar * (beta ** m) * g, prob.B_tot, prob.x_u)
                m += 1
                if f(z) - f(x_s) >= sigma * np.dot(g, (z - x_s)):
                    break
            z = x_s
            f_current = f(z)

            print(f"n = {n}, f_cur = {f_current}, m = {m}")
        else:
            alpha = 0.05
            z = proj_generalized_simplex_lb(prob.c, z - alpha * g, prob.B_tot, prob.x_u)
            f_current = f(z)
            print(f"n = {n}, f_cur = {f_current}")

        n += 1
    return f_current, z