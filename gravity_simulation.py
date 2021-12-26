import numpy as np

from gravity_force import force_gravity
from potentials import energy_gravity
from simulation import simulate
from post_simulation import save_trajectories
from data_processing import plot_components, plot_single, calculate_kinetic_energy
from statistics import compute_statistics_c
from pressure import calculate_pressure_virial


prefix = "Simulations/GravitySim/"
traj_path = "Simulations/GravitySim/InitialConditions/"
G = 9.81
constants = [G]
dt = 0.0001
M = 1
sim_time = 1.0
equilibration_time = sim_time / 2
Ncube = 128
L = 8
T0 = 1.6

time_step = dt
steps = int(sim_time / time_step)

grid, pos, vels, E_pot = simulate(
    Ncube=Ncube,
    T0=T0,
    L=L,
    M=M,
    steps=steps,
    dt=time_step,
    force=force_gravity,
    constants=constants,
    energy=energy_gravity,
    periodic=True,
    from_traj=traj_path,
    heat_bath=False,
    T_heat=None,
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


axis_label = [
    "Time $t$ in $\\left(\\frac{\epsilon}{m\sigma^2}\\right)^{\\frac{1}{2}}$",
    "Pressure $P$ in $\\frac{m}{\\sigma\\tau^2}$",
]
plot_single(
    prefix_data=prefix,
    out_path="Pressure",
    trajectory=pressure / Ncube,
    grid=grid,
    axis_label=axis_label,
)


#compute_statistics_c(ener_traj=E_pot / Ncube, pressure_traj=pressure / Ncube)
