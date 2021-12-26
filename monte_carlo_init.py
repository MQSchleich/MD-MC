import numpy as np
from initialization import InitMonteCarlo, InitVelocity
from potentials import energy_gravity


def init_lj():
    sigma = 0.25
    epsilon = 0.25
    constants = [sigma, epsilon]
    N = 128
    L = 8
    positions = InitMonteCarlo(N, L, constants)
    assert len(positions) == 128, "Added too less particles"
    np.save(arr=positions, file="MCInit/LJ_pos.npy")


def init_gravity(N=128, L=8, T0=2):
    """
    Intialize gravity MC

    Args:
        N (int, optional): [description]. Defaults to 128.
        L (int, optional): [description]. Defaults to 512.
        T0 ([type], optional): [description]. Defaults to 2K.
    """
    G = G = 9.81
    constants = [G]
    positions = InitMonteCarlo(N, L, constants, energy=energy_gravity, tol=10**(3))
    assert len(positions) == N, "Added too less particles"
    np.save(arr=positions, file="Simulations/MCInit/pos_grav.npy")
    vels = InitVelocity(N=N, T0=2)
    assert len(vels) == N, "Added too less particles"
    np.save(arr=vels, file="Simulations/MCInit/vels_grav.npy")


if __name__ == "__main__":
    init_gravity()
