import numpy as np
import matplotlib.pyplot as plt
from numba import njit


def double_log(pos_h, pos_lang, grid, save_path="EquilFeneLangevin/double_log"):
    """Implements the double log plot

    Args:
        pos_h ([type]): [description]
        pos_lang ([type]): [description]
        grid ([type]): [description]
    """
    rmsd_h = rmsd(pos_h[13, :, :])
    rmsd_l = rmsd(pos_lang[13, :, :])
    plt.ylabel("$\\log(t)$")
    plt.xlabel("$\\log(\\text{rmsd})$")
    log_gr = np.log(grid)
    plt.plot(log_gr, np.log(rmsd_h), label="Hamiltonian")
    plt.plot(log_gr, np.log(rmsd_l), label="Langevin")
    plt.xlabel("$\\log(t)$")
    plt.ylabel("$\\log(\\mathrm{rmsd})$")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.show()


def rmsd(particle_traj):
    """calculates the rmsd 

    Args:
        particle_traj ([type]): [description]
    """
    expectation = np.mean(particle_traj, axis=1)
    dx = particle_traj[0, :] - expectation[0]
    dy = particle_traj[1, :] - expectation[1]
    dz = particle_traj[2, :] - expectation[2]

    return (dx ** 2 + dy ** 2 + dz ** 2) ** (1 / 2)


if __name__ == "__main__":
    pos_h = np.load("EquilFenePeriodic/pos.npy")
    pos_l = np.load("EquilFeneLangevin/pos.npy")
    grid = np.load("EquilFeneLangevin/grid.npy")
    double_log(pos_h=pos_h, pos_lang=pos_l, grid=grid)
