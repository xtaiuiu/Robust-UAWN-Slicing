from network_classes.network_slice import Slice
from scenarios.UE_creators import create_UE_set
import numpy as np


def create_slice(n_UEs, tilde_R_l, tilde_R_u, b_width, network_radius):
    UE_set = create_UE_set(n_UEs, network_radius, tilde_R_l, tilde_R_u)
    return Slice(UE_set, b_width)


def create_slice_set(n_slices, network_radius, n_UEs_per_slice):
    # create a randomly generated slice set
    slice_set = []
    embb_bandwidths = np.array([0.1, 0.2, 0.5])
    urllc_bandwidths = np.array([0.05, 0.1])
    # bandwidths = np.array([1])  # for test only
    # num_UEs = np.array([1])
    for i in range(n_slices):
        slice_type = np.random.choice(['embb', 'urllc'])
        if slice_type == 'embb':
            b_width = np.random.choice(embb_bandwidths)
            tilde_R_l, tilde_R_u = 0.5, 2
        else:  # urllc slice
            b_width = np.random.choice(urllc_bandwidths)
            tilde_R_l, tilde_R_u = 0.1, 0.5
        slice_set.append(create_slice(n_UEs_per_slice, tilde_R_l, tilde_R_u, b_width, network_radius))
    return slice_set
