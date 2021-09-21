from matplotlib.pyplot import axis
import numpy as np
from numba import njit

def energy_lj_plot_1(distances, constants, box_length):
    """
    """
    epsilon, sigma = constants
    tmp = sigma / distances
    tmp_six = tmp ** 6
    tmp_twelve = tmp_six ** 2
    e_pot = 4 * epsilon * (tmp_twelve - tmp_six)
    # e_pot[np.isnan(e_pot)] = 0

    return e_pot

def energy_lj(positions, constants, box_length):
    """
    """
    for i in range(0,num-1): 
        for j in range(i+1, num): 
            difference=positions[j,:]- positions[i,:]
            distance = np.linalg.norm(difference)
            if distance> box_length/2: 
                 distance -= box_length/2
            elif distance<= -box_length/2: 
                distance += box_length/2 
                
#TODO implement
def energy_lj_fast(positions, constants, box_length):
    """

    :param positions:
    :type positions:
    :param constants:
    :type constants:
    :param box_length:
    :type box_length:
    :return:
    :rtype:
    """
    epsilon, sigma = constants
    separations = positions[:, None, :] - positions

    # check periodic boundary conditions
    separations[separations > box_length / 2] -= box_length
    separations[separations <= -box_length / 2] += box_length

    # calculate NxN matrix with distances |r[i] - r[j]|
    # set zero values to None for calculation of acceleration

    distances = np.linalg.norm(separations, axis=-1)
    distances[distances == 0] = None

    # calculate potential energy for Lennard Jones potential
    e_pot = (
        4
        * epsilon
        * (np.power((distances / sigma), -12) - np.power((distances / sigma), -6))
    )
    e_pot[np.isnan(e_pot)] = 0

    return np.sum(e_pot)


def kinetic_energy(V, M):
    """Calculates the kinetic energy 

    Args:
        V ([type]): [description]
        M ([type]): [description]
    """

    # normed_vel =np.sum(0.5 * M*V*V,axis=0)
    # assert 1 == 2, str(len(V))
    e_kin = 0
    for i in range(len(V)):
        e_kin += 0.5 * M * np.sum(np.square(V[i, :]))
    return e_kin
