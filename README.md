# The RUNs Framework for Fast and Robust UAV Network Slicing - Simulation Code

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

This repository contains the simulation code for the RUNs Framework UAWN Slicing.



## 🚀 Quick Start
### Installation

#### 1. Install Python 3.10+
If you use conda, run the following command:
```bash
conda create -n runs python=3.10
conda activate runs
```

#### 2. Install dependencies
Run the following command, all dependencies will be installed:
```bash
pip install -e .
```

### Run Simulation
This repository contains has multiple entrypoints to run the simulation.

#### 1. BO-eanbled RUNs framework for 3D UAV deployment and slicing

The following command will generate a four-cell UAWN scenario, and run Bayesian optimization (BO)-eanbled RUNs framework
```bash
python four_cell_scenario.py 
```
You will get the following results.
![Four-cell UAWN scenario](BO_four_cell.png)

![Posterior mean of the Bayesian surrogate model](BO_mean.png)

![Convergence of Bayesian optimization](BO_iteration.png)

#### 2. Comparing RUNs with benchmarks
Run the following command to see our comparisons with existing benchmarks, including SCA, SQP, SHIO, and GBO.
```bash
python compare_with_benchmarks.py 
```
You will get the following results.

#### 3. 2D UAV deployment and slicing
```bash
```

#### 4. Robustness of RUNs
```bash
```

For the detailed design principles of RUNs, please refer to our paper.

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first.