import time

import numpy as np
import cvxpy as cp

from algorithms.MP_Relax_algorithms.benchmark_algorithms.Lagrange_SQP import Lag_SQP_alg
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.Benchmark_static_power import static_power_alloc
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import DAL_alg
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_x_cvx, optimize_p_BFGS


def sca_majorant_with_backtracking(prob, x0=None, p0=None,
                                   alpha0=1.0, alpha_min=1e-8, beta=0.5,
                                   tol=1e-5, max_iter=200, backtrack_max=20,
                                   eps_diag=1e-9, solver=cp.SCS, verbose=False):
    """
    SCA 主循环（严格 majorant + backtracking），采用用户提供的 quadratic upper bound:
       sum_n 0.5*( sqrt(p^k_n/x^k_n) * x_n^2 + sqrt(x^k_n/p^k_n) * p_n^2 ) <= P

    输入:
      prob: OptimizationProblem 实例（拥有 a,b,x_u,c,B_tot,P,p_u 属性）
      x0,p0: 初始可行点（若 None 则用简单启发式初始化）
      alpha0: 初始步长
      alpha_min: 最小步长（回溯停止阈值）
      beta: 回溯缩放因子 (0<beta<1)
      tol: 变量变化收敛容差（L2）
      max_iter: 最大 SCA 迭代次数
      backtrack_max: 每轮最多回溯次数
      eps_diag: 防止除 0 的小常数（用于 sqrt(p^k/x^k) 等）
      solver: cvxpy solver for QP with quadratic constraint (default SCS)
      verbose: 是否打印中间信息

    返回:
      phi_best: float, 最大化目标 phi(x,p) 的值（在收敛点）
      xk, pk: ndarray, 收敛的变量
      k: int, 实际迭代次数
    """

    # unpack problem data
    a = np.asarray(prob.a, dtype=float)
    b = np.asarray(prob.b, dtype=float)
    x_u = np.asarray(prob.x_u, dtype=float)
    c = np.asarray(prob.c, dtype=float)
    B_tot = float(prob.B_tot)
    P = float(prob.P)
    p_u = np.asarray(prob.p_u, dtype=float)

    N = a.size
    assert b.size == N and x_u.size == N and p_u.size == N and c.size == N

    # helper: objective (maximize phi)
    def phi(x, p):
        return float(np.sum(a * x * np.log1p(b * p)))

    # helper: G = -phi (we minimize G)
    def G(x, p):
        return -phi(x, p)

    # gradient of G (w.r.t x and p)
    def grad_G(x, p):
        denom = 1.0 + b * p
        denom = np.maximum(denom, 1e-12)
        grad_x = - a * np.log1p(b * p)         # dG/dx
        grad_p = - a * x * (b / denom)         # dG/dp
        return np.concatenate([grad_x, grad_p])

    # pack/unpack helpers
    def pack(x, p):
        return np.concatenate([x, p])
    def unpack(u):
        return u[:N].copy(), u[N:].copy()

    # initialization (feasible)
    if x0 is None:
        # simple feasible initial x: allocate proportionally to 1/c then ensure >= x_u
        w = np.maximum(c, 1e-12)
        x_init = (B_tot / np.sum(w)) * (1.0 / w)
        x_init = np.maximum(x_init, x_u)
        # if still violates due to x_u, scale down
        if (c @ x_init) > B_tot:
            # scale down towards x_u
            surplus = (c @ x_init) - B_tot
            # simple fallback: set x = x_u and try uniform remainder (if any)
            x_init = x_u.copy()
            free = B_tot - (c @ x_init)
            if free > 0:
                # distribute free proportional to 1/c
                add = free * (1.0 / w) / np.sum(1.0 / w)
                x_init += add
        x0 = x_init

    if p0 is None:
        # feasible p such that x^T p <= P; simple choose p = max(p_u, P / sum(x0) )
        p_init = np.maximum(p_u, (P / (np.sum(x0) + 1e-12)) * np.ones(N))
        p0 = p_init

    xk = x0.copy()
    pk = p0.copy()

    alpha = float(alpha0)

    for k in range(1, max_iter + 1):
        uk = pack(xk, pk)
        Gk = G(xk, pk)
        gradk = grad_G(xk, pk)   # length 2N

        accepted = False
        alpha_local = alpha
        backtries = 0
        last_ucand = None

        # prepare safe coefficients for quadratic upper bound:
        # a_n = sqrt(p_n^k / x_n^k), b_n = sqrt(x_n^k / p_n^k)
        # ensure denominator not zero
        xk_safe = np.maximum(xk, eps_diag)
        pk_safe = np.maximum(pk, eps_diag)
        coef_x = np.sqrt(pk_safe / xk_safe)
        coef_p = np.sqrt(xk_safe / pk_safe)

        while alpha_local >= alpha_min and backtries < backtrack_max:
            # gradient step center
            v = uk - alpha_local * gradk
            v_x = v[:N]; v_p = v[N:]

            # solve projection QP: min ||x - v_x||^2 + ||p - v_p||^2
            # s.t. x >= x_u, p >= p_u, c^T x <= B_tot,
            #      sum_n 0.5*( coef_x[n] * x_n^2 + coef_p[n] * p_n^2 ) <= P
            x_var = cp.Variable(N)
            p_var = cp.Variable(N)

            objective = cp.Minimize(cp.sum_squares(x_var - v_x) + cp.sum_squares(p_var - v_p))
            constraints = [
                x_var >= x_u,
                p_var >= p_u,
                c @ x_var <= B_tot,
                0.5 * cp.sum(cp.multiply(coef_x, cp.square(x_var)) + cp.multiply(coef_p, cp.square(p_var))) <= P
            ]

            subprob = cp.Problem(objective, constraints)
            # solve QP with quadratic constraint; SCS can handle second-order cones/quadratic constraints
            try:
                subprob.solve(solver=solver, warm_start=True, verbose=False)
            except Exception as e:
                # solver failure: shrink alpha and retry
                if verbose:
                    print(f"[iter {k}] solver failed on backtrack {backtries}: {e}; shrink alpha")
                alpha_local *= beta
                backtries += 1
                continue

            if subprob.status not in ["optimal", "optimal_inaccurate"]:
                # shrink and retry
                if verbose:
                    print(f"[iter {k}] subproblem status {subprob.status}; shrink alpha")
                alpha_local *= beta
                backtries += 1
                continue

            x_cand = x_var.value
            p_cand = p_var.value
            ucand = pack(x_cand, p_cand)
            last_ucand = ucand

            # check majorant inequality:
            lhs = G(x_cand, p_cand)
            rhs = Gk + gradk.dot(ucand - uk) + 0.5 * (1.0 / alpha_local) * np.linalg.norm(ucand - uk)**2

            if lhs <= rhs + 1e-8:
                accepted = True
                break
            else:
                # shrink alpha and retry
                if verbose:
                    print(f"[iter {k}] backtrack {backtries}: majorant failed (lhs={lhs:.6e} rhs={rhs:.6e}), shrink alpha")
                alpha_local *= beta
                backtries += 1

        if not accepted:
            # if we failed to find acceptable alpha, accept last candidate (or break)
            if last_ucand is None:
                # cannot make progress -> terminate
                if verbose:
                    print(f"[iter {k}] backtracking failed with no feasible candidate; terminating")
                break
            else:
                # accept last candidate but warn
                if verbose:
                    print(f"[iter {k}] accepting last candidate after backtracking (alpha_local={alpha_local:.2e})")
                xk, pk = unpack(last_ucand)
                # decrease alpha for next iter
                alpha = max(alpha_local * beta, alpha_min)
        else:
            # accept candidate
            xk, pk = x_cand, p_cand
            # optional: try to increase alpha mildly (be more aggressive next time)
            alpha = min(alpha0, alpha_local / beta)

        # stopping check: use norm of change
        diff = (np.linalg.norm(pack(xk, pk) - uk))/(1 + np.linalg.norm(uk))
        if verbose:
            print(f"Iter {k}: phi={phi(xk,pk):.6f}, G={G(xk,pk):.6e}, diff={diff:.4e}, alpha_local={alpha_local:.2e}")

        if diff < tol:
            if verbose:
                print(f"Converged at iter {k} with diff {diff:.2e}")
            break

    phi_best = phi(xk, pk)
    return -phi_best, xk, pk, k


if __name__ == '__main__':
    from algorithms.MP_Relax_algorithms.MP_Relax_problem import create_optimization_instance

    # np.random.seed(0)
    prob = create_optimization_instance(n=50)
    t = time.perf_counter()
    # use static allocation as a warm start
    f_static, x_static, p_static = static_power_alloc(prob)

    f_opt, x_opt, p_opt, iters = sca_majorant_with_backtracking(prob, x0=prob.x_u, p0=prob.p_u, tol=1e-4, verbose=True)
    print("Optimal value:", f_opt)
    # print("x* =", x_opt)
    # print("p* =", p_opt)
    print("iterations:", iters)
    print("execution time: ", time.perf_counter() - t)

    t = time.perf_counter()
    f, x, y, n = Lag_SQP_alg(prob, prob.x_u, prob.p_u)
    print(f"finished in {time.perf_counter() - t} sec, f = {f: .8f}")

    t = time.perf_counter()
    # use static allocation as a warm start
    f_static, x_static, p_static = static_power_alloc(prob)
    f, x, y, _, _, _, _ = DAL_alg(prob, x_static, p_static, subx_optimizer=optimize_x_cvx,
                                  subp_optimizer=optimize_p_BFGS)
    print(f"finished in {time.perf_counter() - t} sec, f = {f: .8f}")

