import logging
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
from utils.logger import get_logger

LOG_LEVEL = logging.WARNING  # Change to DEBUG for more verbose output
logger = get_logger(__name__)

class MyBayesianOptimization(BayesianOptimization):
    # an override BO class, which only alters the plotting functions
    def plot_convergence(self, filename=None):
        '''
        Plots to evaluate the convergence of standard Bayesian optimization algorithms
        '''
        plt.rcParams.update({'font.size': 18})
        Xdata, best_Y = self.X, self.Y_best
        n = Xdata.shape[0]
        plt.figure(figsize=(8, 6))
        # Estimated m(x) at the proposed sampling points
        plt.plot(list(range(n)), -best_Y, '-o')
        # plt.title(r'Convergence of $BO$')
        plt.xlabel('Iteration')
        plt.ylabel('Total data rate (Mbps)')
        grid(True)

        if filename != None:
            savefig(filename)
        else:
            plt.show()


    def plot_mean(self, filename=None):
        # plot the posterior mean of the surrogate model
        pass
        # n = self.X.shape[0]
        # colors = np.linspace(0, 1, n)
        # cmap = plt.cm.Reds
        # norm = plt.Normalize(vmin=0, vmax=1)
        #
        # # define plot func
        # points_var_color = lambda X: plt.scatter(X[:, 0], X[:, 1], c=colors, label=u'observations', cmap=cmap,
        #                                          norm=norm, s=50)
        # points_one_color = lambda X: plt.plot(X[:, 0], X[:, 1], 'r.', markersize=10, label=u'observations')
        #
        # # generate grid
        # X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
        # X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
        # x1, x2 = np.meshgrid(X1, X2)
        # X = np.hstack((x1.reshape(200 * 200, 1), x2.reshape(200 * 200, 1)))
        # acqu = acquisition_function(X)
        # acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))
        # acqu_normalized = acqu_normalized.reshape((200, 200))
        # m, v = model.predict(X)
        # m = -m
        #
        # # ===== 1. Posterior mean =====
        # plt.figure()
        # contour_mean = plt.contourf(X1, X2, m.reshape(200, 200), 100)
        # plt.colorbar(contour_mean, ticks=np.linspace(-2, 1, 4), format='%d')
        # if color_by_step:
        #     points_var_color(Xdata)
        # else:
        #     points_one_color(Xdata)
        # plt.xlabel(label_x)
        # plt.ylabel(label_y)
        # # plt.title('Posterior mean')
        # plt.axis((bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]))
        # # plt.gca().set_aspect('equal')
        # plt.legend()
        # if filename:
        #     plt.savefig(f"{filename}_mean.png", bbox_inches='tight')
        # plt.show()
        #
        # # ===== 新增：Posterior mean (3D Mesh) =====
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(
        #     x1, x2, m.reshape(200, 200),  # X1, X2, Z
        #     cmap='viridis',
        #     rstride=5, cstride=5,
        #     alpha=0.8,
        #     linewidth=0,
        #     antialiased=True
        # )
        # ax.scatter(
        #     Xdata[:, 0], Xdata[:, 1], -m.min(),
        #     c='red', marker='o', s=50, label='Observations'
        # )
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.set_xlabel('')
        # ax.set_ylabel('')
        # ax.set_zlabel('')
        # ax.set_title('')
        # ax.grid(True)
        # if filename:
        #     plt.savefig(f"{filename}_mean_3d.png", bbox_inches='tight')
        # plt.show()
        #
        # # ===== 2. Posterior std =====
        # plt.figure()
        # contour_std = plt.contourf(X1, X2, np.sqrt(v.reshape(200, 200)), 100)
        # plt.colorbar(contour_std, ticks=np.linspace(0, 0.3, 4), format='%.1f')
        # if color_by_step:
        #     points_var_color(Xdata)
        # else:
        #     points_one_color(Xdata)
        # plt.xlabel(label_x)
        # plt.ylabel(label_y)
        # # plt.title('Posterior sd.')
        # plt.axis((bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]))
        # # plt.gca().set_aspect('equal')
        # plt.legend()
        # if filename:
        #     plt.savefig(f"{filename}_sd.png", bbox_inches='tight')
        # plt.show()
        # if filename != None:
        #     savefig(filename)
        # else:
        #     plt.show()

    def plot_std(self, filename=None):
        # plot the posterior standard deviation of the surrogate model
        if filename != None:
            savefig(filename)
        else:
            plt.show()

    def plot_contour(self, filename=None):
        # plot the contour of the surrogate model
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
    print(f" Bayesian optimization started...")
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
    # myProblem.save_report('saved_report.txt')
    # myProblem.save_evaluations("saved_evaluations.csv")
    myProblem.plot_acquisition(label_x="x", label_y="y")
    myProblem.plot_convergence(filename=filename)
    plt.show()

    # print(f"runtime without X and Y in {time.perf_counter() - t} seconds")

    print(f"NOMA: f_opt = {-myProblem.fx_opt}, x_opt = {myProblem.x_opt}")
    return -myProblem.fx_opt, myProblem.x_opt



if __name__ == "__main__":
    logging.disable(logging.INFO)

    #np.random.seed(0)
    sc = create_scenario(2, 100)
    # sc.plot_scenario_kde()
    # sc.reset_scenario()
    #
    # sc.uav.u_x, sc.uav.u_y = sc.get_UEs_center()
    # f_center = bcd(sc)  # the objective function at the center of the points.
    #
    # sc.uav.u_x, sc.uav.u_y = 0, 0
    # f_cell_center = bcd(sc)  # the objective function at the center of the cell.
    f_Bayes, x_Bayes = optimize_horizontal_Bayesian(sc)
    # print(f"f_center = {f_center}, f_Bayes = {f_Bayes}, diff = {f_center - f_Bayes}")
    # print(f"f_cell_center = {f_cell_center}, f_Bayes = {f_Bayes}, diff = {f_cell_center - f_Bayes}")
    # print(f"x_center = {sc.get_UEs_center()}, x_Bayes = {x_Bayes}")
