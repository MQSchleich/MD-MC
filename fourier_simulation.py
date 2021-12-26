import numpy as np
import matplotlib.pyplot as plt


def calculate_fourier_energy(num_particles, n):
    """Calculates the Fourier Spectrum

    Args:
        n ([type]): [description]
    """
    N_freqs = np.linspace(0, n, n + 1)
    energies = -num_particles / 2 * (1 + N_freqs ** 2)
    plt.plot(1 / energies)
    plt.show()


def calculate_fourier_traj(num_particles, t, k):
    """
    """


if __name__ == "__main__":
    calculate_fourier_spec(num_particles=128, n=1000)
