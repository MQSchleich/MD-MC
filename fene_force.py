import numpy as np
from numpy.lib.function_base import _diff_dispatcher
from numba import njit


@njit
def fene_chain_force(positions, constants=np.array([1, 1])):
    """Calculates the fene force for a chain

    Args:
        positions ([type]): [description]
        constants ([type]): [description]
        box_length ([type]): [description]

    Returns:
        [type]: [description]
    """
    r_max, K = constants
    num = positions.shape[0]
    force = np.zeros((num, num, 3))
    for i in range(0, num - 1):
        for j in range(i + 1, i + 2):
            difference = positions[j, :] - positions[i, :]
            distance = np.linalg.norm(difference)
            force[i, j, :] = (
                (-K * r_max ** 2 * distance)
                / (distance ** 2 - r_max ** 2)
                * (difference / distance)
            )
            force[j, i, :] -= force[i, j, :]

    return np.sum(force, axis=1)


@njit
def force_fene_periodic(positions, box_length, r_max=1, K=15):
    """Calculates the fene force for a chain

    Args:
        positions ([type]): [description]
        constants ([type]): [description]
        box_length ([type]): [description]

    Returns:
        [type]: [description]
    """
    num = positions.shape[0]
    force = np.zeros((num, num, 3))
    box_length = num * r_max / 3
    cond = box_length / 2.0
    for i in range(0, num - 1):
        for j in range(i + 1, i + 2):
            difference = positions[j, :] - positions[i, :]
            difference_x = difference[0]
            difference_y = difference[1]
            difference_z = difference[2]
            if difference_x > cond:
                difference_x -= cond
            elif difference_x <= -box_length / 2:
                difference_x += box_length / 2
            if difference_y > box_length / 2:
                difference_y -= box_length / 2
            elif difference_y <= -box_length / 2:
                difference_y += box_length / 2
            if difference_z > box_length / 2:
                difference_z -= box_length / 2
            elif difference_z <= -box_length / 2:
                difference_z += box_length / 2

            distance = np.sqrt(
                difference_x ** 2 + difference_y ** 2 + difference_z ** 2
            )
            force[i, j, :] = (
                (-K * r_max ** 2 * distance)
                / (distance ** 2 - r_max ** 2)
                * (difference / distance)
            )
            force[j, i, :] -= force[i, j, :]
    difference = positions[-1, :] - positions[1, :]
    difference_x = difference[0]
    difference_y = difference[1]
    difference_z = difference[2]
    if difference_x > box_length / 2:
        difference_x -= box_length / 2
    elif difference_x <= -box_length / 2:
        difference_x += box_length / 2
    if difference_y > box_length / 2:
        difference_y -= box_length / 2
    elif difference_y <= -box_length / 2:
        difference_y += box_length / 2
    if difference_z > box_length / 2:
        difference_z -= box_length / 2
    elif difference_z <= -box_length / 2:
        difference_z += box_length / 2
    distance = np.sqrt(difference_x ** 2 + difference_y ** 2 + difference_z ** 2)
    force[1, -1, :] = (
        (-K * r_max ** 2 * distance)
        / (distance ** 2 - r_max ** 2)
        * (difference / distance)
    )
    force[-1, 1, :] -= force[1, -1, :]
    return np.sum(force, axis=1)


if __name__ == "__main__":
    positions = np.random.random((4, 3))
    force = force_fene_periodic(positions, box_length=48 * 1 / 3)
    print(positions)
