import numpy as np
from data_processing import calculate_kinetic_energy


def calculate_pressure(vel_trajectory, masses, N, k, epsilon, sigma_q=512):
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


def calculate_pressure_virial(k, N, volume, kinetic_energy):
    """

    :return:
    """
    T = kinetic_energy / k * 2 / 3
    p = (k * T * N) / (volume) - 1 / (3 * volume) * kinetic_energy
    return p
