from network_classes.network_slice import Slice
from scenarios.UE_creators import create_UE_set
import numpy as np


def create_slice(n_UEs, tilde_R_l, tilde_R_u, b_width, network_radius, r_sla):
    UE_set = create_UE_set(n_UEs, network_radius, tilde_R_l, tilde_R_u)
    return Slice(UE_set, b_width, r_sla)


def create_slice_set(n_slices, network_radius, n_UEs_per_slice):
    # create a randomly generated slice set
    slice_set = []
    embb_bandwidths = np.array([0.1, 0.2, 0.5])
    urllc_bandwidths = np.array([0.01, 0.5])
    # bandwidths = np.array([1])  # for test only
    # num_UEs = np.array([1])
    for i in range(n_slices):
        slice_type = np.random.choice(['embb', 'urllc'])
        if slice_type == 'embb':
            b_width = np.random.choice(embb_bandwidths)
            r_sla = np.random.uniform(0.1, 1)
        else:  # urllc slice
            b_width = np.random.choice(urllc_bandwidths)
            r_sla = np.random.uniform(0.01, 0.05)
        tilde_R_l, tilde_R_u = 1, 3  # \zeta_{ij} in the paper
        slice_set.append(create_slice(n_UEs_per_slice, tilde_R_l, tilde_R_u, b_width, network_radius, r_sla))
    return slice_set
