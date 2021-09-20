import numpy as np
from data_processing import calculate_kinetic_energy


def calculate_pressure_virial(vel_trajectory, masses, N, k, epsilon, sigma_q=512):
    """calculates the pressure according to the virial theorem

    Args:
        vel_trajectory ([type]): [description]
        masses ([type]): [description]
        N ([type]): Particle number
        k ([type]): Boltzman constant
sigma_q (int, optional): [description]. Defaults to 512.
    """
    e_kin = calculate_kinetic_energy(vel_trajectory=vel_trajectory, mass=masses)
    T = e_kin / k * 2 / 3
    p = (k * T * N) / (sigma_q) + 2 / (3 * sigma_q) * e_kin
    return p


def calculate_pressure_vdw(vel_trajectory, masses, N, k, epsilon, sigma_q=512):
    """
    Calculate the thermodynamic pressure according to the thermodynamic temperature
    :param position:
    :param constants:
    :param masses:
    :return:
    """
    e_kin = calculate_kinetic_energy(vel_trajectory=vel_trajectory, mass=masses)
    T = (2 * e_kin) / (3 * k)
    return calculate_analytical_pressure(N, k, T, epsilon, sigma_q)


def calculate_analytical_pressure(N, k, T, epsilon, sigma_q=512):
    """
    Calculates the expecatation of the pressure according to the vdW equation
    :return:
    """
    a = (8 * np.pi * epsilon * sigma_q) / (3)
    b = (2 * np.pi * sigma_q) / (3)
    p = (N * k * T) / (sigma_q - N * b) - a * N ** 2
    return p
