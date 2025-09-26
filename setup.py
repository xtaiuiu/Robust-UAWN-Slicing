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
    python_requires='>=3.12',
    install_requires=[
        "numpy==1.26.4",
        "mealpy==3.0.1",
        "scipy==1.16.1",
        "matplotlib==3.9.1",
        "pandas==2.2.2",
        "cvxpy==1.5.2",
        "GPyOpt==1.2.6",
        "GPy==1.10.0",
        "openpyxl==3.1.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.12.4",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
