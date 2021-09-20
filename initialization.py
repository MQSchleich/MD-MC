from operator import pos
import numpy as np
import math
import openmc
import matplotlib.pyplot as plt
from fene_potential import fene_chain_potential
from potentials import energy_lj_fast
from tqdm import tqdm
from numba import njit


@njit
def InitFeneChain(N, L, constants):
    """[summary]

    Args:
        N ([type]): [description]
        L ([type]): [description]
        constants ([type]): [description]
        tol ([type], optional): [description]. Defaults to 10**(3).

    Returns:
        [type]: [description]
    """
    r_max, K = constants
    N_cube  = math.floor(N**(1/3))+1
    pos = InitPositionCubic(N, L=N*r_max/3)
    return pos[:N, :]



def InitMonteCarlo(N, L, constants, tol=10 ** (3)):
    """intialize particles with MC 

    Args:
        N ([type]): [description]
        L ([type]): [description]
    """
    epsilon, sigma = constants
    N_cube = math.floor(N ** (1 / 3))
    positions = InitPositionCubic(Ncube=N_cube, L=L)[:2, :]
    # assert positions.shape == (125,3)
    counter = 0
    prev_energy = energy_lj_fast(positions, constants, box_length=L)
    for i in tqdm(range(126)):

        energy_high = True
        while energy_high == True:
            ran_floats = np.random.uniform(low=-4, high=4, size=3).reshape((1, 3))
            tmp_new_pos = np.concatenate((positions, ran_floats), axis=0)
            new_energy = energy_lj_fast(positions, constants, box_length=L)
            dif = abs(prev_energy - new_energy)
            if dif <= tol:
                positions = tmp_new_pos
                prev_energy = new_energy
                counter += 1
                print("the Monte Carlo simulation stopped at tol=", dif)
                print("added particle ", counter)
                energy_high = False
    return positions


def InitPositionCircles(N, L):
    """generates cubic initialization exam

    Args:
        N ([type]): [description]
        L ([type]): [description]
    """

    T = [1, 7, 13]
    R = [0.0, 2, 4]

    particles = np.zeros((128, 3))

    def rtpairs(r, n):

        for i in range(len(r)):
            for j in range(n[i]):
                yield r[i], j * (2 * np.pi / n[i])

    counter = 0
    for i in range(4):
        for j in range(len(R)):
            for k in range(len(T)):
                r = R[j]
                t = j * (2 * np.pi / T[k])
                particles[counter, :] = [r * np.cos(t), r * np.sin(t), i]
                counter += 1
    return particles


def InitPositionCubicMC(N, L):
    min_x = openmc.XPlane(x0=-4, boundary_type="reflective")
    max_x = openmc.XPlane(x0=4, boundary_type="reflective")
    min_y = openmc.YPlane(y0=-4, boundary_type="reflective")
    max_y = openmc.YPlane(y0=4, boundary_type="reflective")
    min_z = openmc.ZPlane(z0=-4, boundary_type="reflective")
    max_z = openmc.ZPlane(z0=4, boundary_type="reflective")
    region = +min_x & -max_x & +min_y & -max_y & +min_z & -max_z
    positions = openmc.model.pack_spheres(
        radius=0.4,
        region=region,
        pf=None,
        num_spheres=128,
        contraction_rate=0.001,
        seed=41,
    )
    return positions

@njit
def InitPositionCubic(Ncube, L):
    """Places Ncube^3 atoms in a cubic box; returns position vector"""
    N = Ncube ** 3
    position = np.zeros((N, 3))
    rs = L / Ncube
    roffset = L / 2 - rs / 2
    n = 0
    # Note: you can rewrite this using the `itertools.product()` function
    for x in range(0, Ncube):
        for y in range(0, Ncube):
            for z in range(0, Ncube):
                if n < N:
                    position[n, 0] = rs * x - roffset
                    position[n, 1] = rs * y - roffset
                    position[n, 2] = rs * z - roffset
                n += 1
    return position

@njit
def InitVelocity(N, T0, mass=1.0, seed=1):
    dim = 3
    np.random.seed(seed)
    # generate N x dim array of random numbers, and shift to be [-0.5, 0.5)
    velocity = np.random.random((N, dim)) - 0.5
    sumV = np.sum(velocity, axis=0) / N  # get the average along the first axis
    velocity -= sumV  # subtract off sumV, so there is no net momentum
    KE = np.sum(velocity * velocity)  # calculate the total of V^2
    vscale = np.sqrt(dim * N * T0 / (mass * KE))  # calculate a scaling factor
    velocity *= vscale  # rescale
    return velocity

def visualize_init(positions): 
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], marker="o")
    ax.set_xlabel("X coordinate $\\frac{l_x}{\\sigma}$")
    ax.set_ylabel("Y coordinate $\\frac{l_y}{\\sigma}$")
    ax.set_zlabel("Z coordinate $\\frac{l_z}{\\sigma}$")

    plt.show()


if __name__ == "__main__":
    K = 1
    r_max = 1
    constants = [K, r_max]
    N = 48
    L = N*r_max/3
    positions = InitFeneChain(N, L, constants)

    visualize_init(positions)
    pos_2 = np.load("EquilFeneChain/pos.npy")
    visualize_init(pos_2[:,:,0])
    
