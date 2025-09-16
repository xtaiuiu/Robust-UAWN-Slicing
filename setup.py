from setuptools import setup

setup(
    name='RUNs_UAWN_slicing',
    version='1.0',
    packages=['test', 'test.bcd_sum_test', 'test.rounding_test', 'test.projection_test', 'utils',
              'utils.gradient_projection_methods', 'scenarios', 'algorithms', 'algorithms.rounding',
              'algorithms.MP_Relax_algorithms', 'algorithms.MP_Relax_algorithms.main_algorithm',
              'algorithms.MP_Relax_algorithms.main_algorithm.BCD', 'algorithms.MP_Relax_algorithms.main_algorithm.DAL',
              'algorithms.MP_Relax_algorithms.main_algorithm.Sub_xp',
              'algorithms.MP_Relax_algorithms.main_algorithm.Bayesian_opt',
              'algorithms.MP_Relax_algorithms.benchmark_algorithms', 'network_classes', 'experiment_rounding',
              'experiment_comparison', 'experiment_horizontal', 'experiment_robustness', 'experiment_convergence'],
    url='https://scholar.google.com/citations?user=uGLXrecAAAAJ&hl=zh-CN',
    python_requires='>=3.9',
    install_requires=[
        "GPyOpt~=1.2.6",
        "GPy~=1.13.2",
        "numpy",
        "mealpy",
        "scipy",
        "matplotlib",
        "pandas",
        "cvxpy",
    ],
    license='',
    author='Fengsheng Wei',
    author_email='787816998@qq.com',
    description='A fast and robust UAWN network slicing framework.'
)
