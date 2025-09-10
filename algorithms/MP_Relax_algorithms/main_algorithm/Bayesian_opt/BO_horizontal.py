import time

import numpy as np
from GPyOpt.methods import BayesianOptimization
from matplotlib import pyplot as plt
from matplotlib.pyplot import grid, savefig
import GPyOpt

from algorithms.MP_Relax_algorithms.main_algorithm.DAL.Benchmark_static_power import static_power_alloc
from algorithms.MP_Relax_algorithms.main_algorithm.DAL.DAL_algorithm import DAL_alg
from algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp.Solver_algorithm import optimize_x_cvx, optimize_p_BFGS
from scenarios.scenario_creators import scenario_to_problem, create_scenario


class MyBayesianOptimization(BayesianOptimization):
    # an override BO class, which only alters the plotting functions
    def plot_convergence(self, filename=None):
        '''
        Plots to evaluate the convergence of standard Bayesian optimization algorithms
        '''
        Xdata, best_Y = self.X, self.Y_best
        n = Xdata.shape[0]
        plt.figure(figsize=(8, 6))
        # Estimated m(x) at the proposed sampling points
        plt.plot(list(range(n)), -best_Y, '-o')
        plt.title(r'Convergence of $B^3CD$')
        plt.xlabel('Iteration')
        plt.ylabel('Best Objective Value')
        grid(True)

        if filename != None:
            savefig(filename)
        else:
            plt.show()

def optimize_horizontal_Bayesian(sc, eps=1e-6, filename=None):
    def DAL_wrapper(X):
        # a wrapper function for the DAL algorithm. As the GPyOpt library expects a function of the form f(x),
        # where X is an 2-dimensional array, and this library does not accept functions with extra arguments,
        # thus we use a wrapper

        # Initialize output array with same number of rows as input X
        results = np.zeros((X.shape[0], 1))

        # Process each input point
        for i in range(X.shape[0]):
            x = X[i, 0]  # x-coordinate
            y = X[i, 1]  # y-coordinate
            # load the scenario
            # sc.reset_scenario()
            sc.uav.x, sc.uav.y = x, y
            prob = scenario_to_problem(sc)
            f_static, x_static, p_static = static_power_alloc(prob)
            f, _, _, _, _, _, _ = DAL_alg(prob, x_static, p_static, subx_optimizer=optimize_x_cvx,
                                          subp_optimizer=optimize_p_BFGS)
            results[i, 0] = f  # store the negative value of the objective function

        return results  # return 2D array of results

    R = sc.pn.radius
    bounds = [{'name': 'x', 'type': 'continuous', 'domain': (-R, R)},
              {'name': 'y', 'type': 'continuous', 'domain': (-R, R)}]

    t = time.perf_counter()
    # Get center coordinates and create initial points array
    X = np.array([
#        [0, 0],                         # Center of cell
        [R/2, R/2],                     # Top-right quadrant point
        [-R/2, R/2],                    # Top-left quadrant point
        [R/2, -R/2],                    # Bottom-right quadrant point
        [-R/2, -R/2],                    # Bottom-left quadrant point
        [R/4, R/4],                     # Top-right quarter point
        [-R/4, R/4],                    # Top-left quarter point
        [R/4, -R/4],                    # Bottom-right quarter point
        [-R/4, -R/4],                   # Bottom-left quarter point
    ])
    Y = DAL_wrapper(X)
    #print(Y)
    t = time.perf_counter()
    myProblem = MyBayesianOptimization(DAL_wrapper, bounds, X=X, Y=Y, normalize_Y=True, exact_feval=True)
    myProblem.run_optimization(max_iter=15, eps=1e-8, report_file='report.txt')
    print(f"runtime with X and Y in {time.perf_counter() - t} seconds")
    myProblem.save_report('saved_report.txt')
    myProblem.save_evaluations("saved_evaluations.csv")
    myProblem.plot_acquisition(label_x="x", label_y="y")
    myProblem.plot_convergence(filename=filename)
    plt.show()

    # print(f"runtime without X and Y in {time.perf_counter() - t} seconds")

    print(f"NOMA: f_opt = {-myProblem.fx_opt}, x_opt = {myProblem.x_opt}")
    return -myProblem.fx_opt, myProblem.x_opt



if __name__ == "__main__":
    #np.random.seed(0)
    sc = create_scenario(50, 100)
    # sc.plot_scenario_kde()
    # sc.reset_scenario()
    #
    # sc.uav.u_x, sc.uav.u_y = sc.get_UEs_center()
    # f_center = bcd(sc)  # the objective function at the center of the points.
    #
    # sc.uav.u_x, sc.uav.u_y = 0, 0
    # f_cell_center = bcd(sc)  # the objective function at the center of the cell.
    f_Bayes, x_Bayes = optimize_horizontal_Bayesian(sc, filename='convergence_curve.png')
    # print(f"f_center = {f_center}, f_Bayes = {f_Bayes}, diff = {f_center - f_Bayes}")
    # print(f"f_cell_center = {f_cell_center}, f_Bayes = {f_Bayes}, diff = {f_cell_center - f_Bayes}")
    # print(f"x_center = {sc.get_UEs_center()}, x_Bayes = {x_Bayes}")
