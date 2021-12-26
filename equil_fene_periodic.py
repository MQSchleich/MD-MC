import numpy as np

from fene_potential import fene_chain_potential, fene_ring_potential
from fene_force import force_fene_periodic
from simulation_fene_five import simulate
from post_simulation import save_trajectories
from data_processing import plot_components, plot_single, calculate_kinetic_energy
from statistics import compute_statistics_c
from pressure import calculate_pressure_virial


prefix = "EquilFenePeriodic/"
traj_path = "EquilFenePeriodic/InitialConditions/"
# pos = np.load(traj_path+"pos.npy")[:,:,-1]
# vels = np.load(traj_path+"vel.npy")[:,:,-1]
r_max = 1.0
K = 15.0
constants = [r_max, K]
dt = 0.002
M = 1
sim_time = 500
equilibration_time = sim_time / 2
Ncube = 48
k_b = 1
L = Ncube * r_max / 3
T0 = 0.1 * K * r_max / k_b

time_step = dt
steps = int(sim_time / time_step)

grid, pos, vels, E_pot = simulate(
    Ncube=Ncube,
    T0=T0,
    L=L,
    M=M,
    steps=steps,
    dt=time_step,
    force=force_fene_periodic,
    energy=fene_ring_potential,
    constants=constants,
    periodic=True,
)
trajs = [grid, pos, vels, E_pot]
save_trajectories(trajs, prefix=prefix)
E_kin = calculate_kinetic_energy(vel_trajectory=vels, mass=M)
E_tot = E_kin + E_pot
E_diff = E_tot - E_tot[0]
axis_label = [
    "Time $t$ in $\\left(\\frac{\epsilon}{m\sigma^2}\\right)^{\\frac{1}{2}}$",
    "Energy difference $\\delta E(t)$ in $\\epsilon$",
]
plot_single(
    prefix_data=prefix,
    out_path="E_diff",
    trajectory=E_diff / Ncube,
    grid=grid,
    axis_label=axis_label,
)

axis_label = [
    "Time $t$ in $\\left(\\frac{\epsilon}{m\sigma^2}\\right)^{\\frac{1}{2}}$",
    "Momentum difference $\\delta p(t)$ in $\\frac{m\\sigma}{\\tau}$",
]
plot_components(
    prefix_data=prefix,
    trajectory=(np.sum(vels, axis=0) / (M * Ncube)),
    grid=grid,
    axis_label=axis_label,
)

axis_label = [
    "Time $t$ in $\\left(\\frac{\epsilon}{m\sigma^2}\\right)^{\\frac{1}{2}}$",
    "Potential Energy $E_{pot}$ in $\\epsilon$",
]
plot_single(
    prefix_data=prefix,
    out_path="E_pot",
    trajectory=E_pot / Ncube,
    grid=grid,
    axis_label=axis_label,
)
