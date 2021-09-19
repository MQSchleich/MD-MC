from post_simulation import save_trajectories
from operator import pos
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from initialization import InitPositionCubic, InitVelocity
from potentials import energy_lj_fast, kinetic_energy
from lj_force import force_lj, force_lj_fast
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
    from_traj=None,
    heat_bath=False,
    initializer=InitPositionCubic,
    T_heat=None,
):
    """Initialize and run a simulation in a Ncube**3 box, for steps"""

    sigma, epsilon = constants
    if from_traj != None:
        positions,velocities = load_initials(from_traj)
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
        if heat_bath == True:
            velocities = InitVelocity(N, T_heat, M)
        E_pot[t] = energy_lj_fast(
            positions, [sigma, epsilon], L
        )  # calculate potential energy contribution
        F = force(
            positions, [sigma, epsilon], L
        )  ## calculate forces; should be a function that returns an N x 3 array
        A = F / M
        nR = VerletNextR(positions, velocities, A, dt)
        # my_pos_in_box(nR, L)  ## from PrairieLearn HW

        nF = force_lj(
            nR, [sigma, epsilon], L
        )  ## calculate forces with new positions nR
        nA = nF / M
        nV = VerletNextV(velocities, A, nA, dt)

        # update positions:
        positions, velocities = nR, nV
        vels[:, :, t] = velocities
        pos[:, :, t] = positions
    grid = np.arange(0, steps * dt, dt)
    return [grid, pos, vels, E_pot]


def VerletNextR(r_t, v_t, a_t, h):
    """Return new positions after one Verlet step"""
    # Note that these are vector quantities.
    # Numpy loops over the coordinates for us.
    r_t_plus_h = r_t + v_t * h + 0.5 * a_t * h * h
    return r_t_plus_h


def VerletNextV(v_t, a_t, a_t_plus_h, h):
    """Return new velocities after one Verlet step"""
    # Note that these are vector quantities.
    # Numpy loops over the coordinates for us.
    v_t_plus_h = v_t + 0.5 * (a_t + a_t_plus_h) * h
    return v_t_plus_h


if __name__ == "__main__":
    sigma = 1.0
    epsilon = 1.0
    constants = [epsilon, sigma]
    import timeit

    dt = 0.001
    M = 1
    sim_time = 10000

    # number of Particles
    Ncube = 4
    N = Ncube ** 3

    # box side length
    L = 8

    # temperature
    T0 = 1.6
    # grid, positions, velocities, energies, E_pot =
    trajs = simulate(
        Ncube,
        T0,
        L,
        M,
        sim_time,
        force=force_lj,
        constants=constants,
        h=dt,
        heat_bath=True,
        T_heat=T0,
    )
    save_trajectories(prefix="IntialConditions/", arrays=trajs)
    velocities = trajs[2]
    E_kin = np.sum(0.5 * M * np.linalg.norm(velocities, axis=1) ** 2, axis=0)
    E_pot = trajs[-1]
    plt.plot(E_kin, label="from_Vels")
    plt.plot(E_pot, label="E_pot")
    plt.plot(E_pot + E_kin, label="E_tot")
    plt.legend()
    plt.show()
