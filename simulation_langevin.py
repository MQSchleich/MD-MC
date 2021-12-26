from post_simulation import save_trajectories
from operator import pos
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

from initialization import InitFeneChain, InitVelocity
from fene_potential import fene_chain_potential
from fene_force import fene_chain_force
from load_data import load_initials


def simulate(
    Ncube,
    T0,
    L,
    M,
    steps,
    dt,
    force,
    constants,
    gamma,
    kbT,
    energy=fene_chain_potential,
    periodic=False,
    from_traj=None,
    pos=None,
    vels=None,
    initializer=InitFeneChain,
):
    """Initialize and run a simulation in a Ncube**3 box, for steps"""

    r_max, K = constants
    if from_traj != None:
        positions = pos
        velocities = vels
        N = len(positions)
    else:
        positions = initializer(Ncube, L, constants)
        N = positions.shape[0]
        velocities = InitVelocity(N, T0, M)

    A = np.zeros((N, 3))
    E_pot = np.zeros(steps)
    vels = np.zeros((N, positions.shape[1], steps))
    pos = np.zeros((N, positions.shape[1], steps))
    for t in tqdm(range(0, steps)):
        E_pot[t] = energy(
            positions, constants
        )  # calculate potential energy contribution
        F = force(
            positions, r_max, K
        )  ## calculate forces; should be a function that returns an N x 3 array
        nA = F / M
        if periodic == True:
            positions[positions >= L / 2] -= L
            positions[positions < -L / 2] += L
        # B
        v = velocity_update(velocities, A=A, dt=dt)
        # A
        nR = VerletNextR(positions, velocities, A, dt)
        # my_pos_in_box(nR, L)  ## from PrairieLearn HW
        # O
        v = random_velocity_update(v=v, gamma=gamma, kBT=kbT, dt=dt)
        # A
        nR = VerletNextR(nR, velocities, A, dt)
        # B
        nF = force(nR, r_max, K)  ## calculate forces with new positions nR
        nA = nF / M
        nV = VerletNextV(velocities, A, nA, dt)

        # update positions:
        positions, velocities = nR, nV
        vels[:, :, t] = velocities
        pos[:, :, t] = positions
    grid = np.arange(0, steps * dt, dt)
    return [grid, pos, vels, E_pot]


@njit
def random_velocity_update(v, gamma, kBT, dt):
    """Implements a random velocity verlet update for the BAOAB scheme


    Returns:
        [type]: [description]
    """
    ran = np.random.normal()
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt(1 - c1 * c1) * np.sqrt(kBT)
    v_new = c1 * v + ran * c2
    return v_new


@njit
def VerletNextR(r_t, v_t, a_t, h):
    """Return new positions after one Verlet step"""
    # Note that these are vector quantities.
    # Numpy loops over the coordinates for us.
    r_t_plus_h = r_t + v_t * h + 0.5 * a_t * h * h
    return r_t_plus_h


@njit
def velocity_update(v, A, dt):
    v_new = v + A * dt / 2.0
    return v_new


@njit
def VerletNextV(v_t, a_t, a_t_plus_h, h):
    """Return new velocities after one Verlet step"""
    # Note that these are vector quantities.
    # Numpy loops over the coordinates for us.
    v_t_plus_h = v_t + 0.5 * (a_t + a_t_plus_h) * h
    return v_t_plus_h
