import numpy as np 
import matplotlib.pyplot as plt

from potentials import energy_lj_plot_1

def plot_lj(num_points, constants, box_length, lj_energy): 
    """plots the lj potential

    Args:
        num_points ([type]): [description]
        lj_energy ([type]): [description]
    """
    epsilon, sigma = constants
    grid = np.linspace(0.01,box_length/2, num=num_points)
    ener = lj_energy(grid, constants, box_length)
    plt.ylim(-epsilon-(epsilon*0.1), box_length/2+0.1*box_length/2)
    plt.plot(grid, ener)
    plt.show()

if __name__ == "__main__": 
    num_points = 10000
    epsilon = 100
    sigma = 0.125
    constants = [epsilon, sigma] 
    box_length = 8
    lj_energy = energy_lj_plot_1
    plot_lj(num_points, constants, box_length, lj_energy)