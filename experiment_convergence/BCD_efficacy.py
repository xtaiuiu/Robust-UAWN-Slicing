# validate the efficacy of BCD
import pandas as pd
from matplotlib import pyplot as plt
from mealpy.math_based import SHIO

from algorithms.MP_Relax_algorithms.main_algorithm.BCD.BCD_algorithm import bcd_al
from algorithms.MP_Relax_algorithms.main_algorithm.BCD.Gradient_descent_BCD import pgd_BCD
from algorithms.MP_Relax_algorithms.main_algorithm.BCD.Heuristic_BCD import BCD_by_heuristic
from scenarios.scenario_creators import load_scenario, scenario_to_problem


def run():
    sc = load_scenario('sc_debug_2.pickle')
    prob = scenario_to_problem(sc)
    lam, rho = 0.01, 100

    # I don't know how to get the iteration history of SHIO, so the iterations are copied
    # from the terminal output

    f, x, p, _, _ = bcd_al(prob, lam, rho, prob.x_u, prob.p_u)

    f, x, p, f_values = pgd_BCD(prob, lam, rho, prob.x_u, prob.p_u)

    model_shio = SHIO.OriginalSHIO(epoch=1000, pop_size=200)
    f_shio, x_shio = BCD_by_heuristic(prob, lam, rho, model_shio)


def load_plot(filename='BCD_efficacy.xlsx'):
    df = pd.read_excel(filename, index_col=0)
    #df = df.iloc[::10]
    plt.rcParams.update({'font.size': 18})

    ax = df.plot(legend=True, lw=2, xlabel=r'Number of iterations', ylabel=r'Function value of $ P_{\lambda, \rho}$',
                 fontsize=16, style=["r-s", "c-->", "m-.d"], markersize=4, grid=True,
                 markerfacecolor='none', markevery=10)
    # ax.set_xticks(G)
    # ax.set_xticklabels([str(i + 6) for i in G], fontsize=16)
    plt.ylim(-3.2, 28.8)
    plt.rcParams['legend.markerscale'] = 2
    ax.legend(['2BCD', 'GPA', 'GA'])
    # plt.legend()

    df = df.iloc[:, [0]]
    df = df[:5]
    plt.rcParams['legend.markerscale'] = 2
    ax = df.plot(legend=False, lw=2,
                 fontsize=32, style=["r-s", "k:^", "m-.d", "c-->"], markersize=10, grid=True,
                 markerfacecolor='none')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    #run()
    load_plot()
