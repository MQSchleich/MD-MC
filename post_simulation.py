import numpy as np


def save_trajectories(arrays, prefix=""):
    """saves the trajectories

    Args:
        prefix ([type]): to path
        arrays ([type]): collection of data points
    """

    grid, positions, velocities, E_pot = arrays

    np.save(arr=grid, file=prefix + "grid.npy")
    np.save(arr=positions, file=prefix + "pos.npy")
    np.save(arr=velocities, file=prefix + "vel.npy")
    np.save(arr=E_pot, file=prefix + "e_pot.npy")
