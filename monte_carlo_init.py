import numpy as np
from initialization import InitMonteCarlo


def init_lj():
    sigma = 0.25
    epsilon = 0.25
    constants = [sigma, epsilon]
    N = 128
    L = 512
    positions = InitMonteCarlo(N, L, constants)
    assert len(positions) == 128, "Added too less particles"
    np.save("MCInit/LJ.npy")


def init_gravity():
    sigma = 0.25
    epsilon = 0.25
    constants = [sigma, epsilon]
    N = 128
    L = 512
    positions = InitMonteCarlo(N, L, constants)
    assert len(positions) == 128, "Added too less particles"
    np.save("MCInit/grav.npy")
