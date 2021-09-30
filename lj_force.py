import numpy as np
from numba import njit

@njit()
def force_lj_fast(positions, constants, box_length):
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
    num = positions.shape[0]
    force = np.zeros((num, num, 3))
    for i in range(0,num-1): 
        for j in range(i+1, num): 
            difference=positions[j,:]- positions[i,:]
            distance = np.linalg.norm(difference)
            if distance> box_length/2: 
                 distance -= box_length/2
            elif distance<= -box_length/2: 
                distance += box_length/2
            lj_part = (sigma/distance)**6
            lj_part_two = lj_part**2
            factor = 24*epsilon
            factor_two = 2*factor
            force[i, j, :] = (factor_two * lj_part_two - factor*lj_part)/distance
            force[j, i, :] -= force[i, j, :]
   
    return np.sum(force, axis=1)


def force_lj(positions, constants, box_length):
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

    idx = np.arange(len(positions))
    pairs = np.meshgrid(idx, idx)

    # calculate NxNx3 array with distances r[i] - r[j]
    separations = positions[pairs[0]] - positions[pairs[1]]

    # check periodic boundary conditions
    separations[separations > box_length / 2] -= box_length
    separations[separations <= -box_length / 2] += box_length

    # calculate NxN matrix with distances |r[i] - r[j]|
    # set zero values to None for calculation of acceleration
    distances = np.linalg.norm(separations, axis=-1)

    # calculate acceleration due to Lennard Jones potential
    acceleration = 48 * epsilon * np.power(sigma, 12) / np.power(
        distances, 13
    ) - 24 * epsilon * np.power(sigma, 6) / np.power(distances, 7)

    # determine unit vectors for the direction of forces
    # calculate forces by acc * unit_vec in every direction
    unit_vectors = separations / distances[:, :, None]
    force = unit_vectors * acceleration[:, :, None]

    # assign None values back to zero
    force = np.nan_to_num(force)

    # return collapsed array -> sum over every component along the right axis
    return np.sum(force, axis=0)
