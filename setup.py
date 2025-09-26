from setuptools import setup, find_packages

setup(
    name='RUNs_UAWN_slicing',
    version='1.0.0',
    description='A fast and robust UAWN network slicing framework.',
    long_description='',
    long_description_content_type='text/markdown',
    url='https://scholar.google.com/citations?user=uGLXrecAAAAJ&hl=zh-CN',
    author='Fengsheng Wei',
    author_email='787816998@qq.com',
    license='',
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    python_requires='>=3.9',
    install_requires=[
        "GPy~=1.13.2",
        "numpy",
        "scipy",
        "matplotlib",
        "cvxpy",
        "pandas",
        "mealpy",
        "setuptools",
        "GPyOpt",
        "pyinstrument",
        "openpyxl",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.12.4",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
