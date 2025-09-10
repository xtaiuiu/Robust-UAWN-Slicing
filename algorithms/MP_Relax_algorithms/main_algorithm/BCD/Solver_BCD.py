from scipy.optimize import minimize
import numpy as np

# 定义目标函数
def objective(x):
    return np.sin(x[0]) + np.cos(x[1])  # 示例非凸函数

def solver_BCD(prob, lam, rho, x_init, p_init):
    l = len(prob.a)
    objective = lambda z: prob.Lagrangian_value(z[:l], z[l:], lam, rho)
    constraints = {'type': 'ineq', 'fun': lambda z: prob.B_tot - np.array([prob.c]) @ z[:l]}
    bounds = [(lb, None) for lb in np.concatenate((prob.x_u, prob.p_u))]
    z0 = np.concatenate((x_init, p_init))
    result = minimize(objective, z0, method='trust-constr', bounds=bounds, constraints=constraints, options={'maxiter': 800000,})
    if result.success:
        return result.fun, result.x[:l], result.x[l:]
    else:
        raise ValueError("solver_BCD failed", result.message)
