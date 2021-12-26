import numpy as np
from numba import njit


@njit()
def force_gravity(positions, constants, box_length):
    """
    Calculates the force for point particles 
    :param positions:
    :type positions:
    :param constants:
    :type constants:
    :param box_length:
    :type box_length:
    :return:
    :rtype:
    """
    num = positions.shape[0]
    force = np.zeros((num, num, 3))
    for i in range(0, num - 1):
        for j in range(i + 1, num):
            difference = positions[j, :] - positions[i, :]
            distance = np.linalg.norm(difference)
            if distance > box_length / 2:
                distance -= box_length / 2
            elif distance <= -box_length / 2:
                distance += box_length / 2
            force[i, j, :] = constants[0] / distance ** 2
            force[j, i, :] -= force[i, j, :]

    return np.sum(force, axis=1)
