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
    python_requires='>=3.10,<3.11',
    install_requires=[
        "numpy==1.26.0",
        "mealpy==2.5.4",
        "scipy==1.12.0",
        "matplotlib==3.8.0",
        "pandas==2.1.1",
        "cvxpy==1.4.2",
        "GPyOpt==1.2.6",
        "GPy==1.10.0",
        "openpyxl==3.1.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
