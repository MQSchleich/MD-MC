from lj_force import force_lj_fast, force_lj
from simulation import simulate
from post_simulation import save_trajectories
from data_processing import plot_single, calculate_kinetic_energy
prefix = "EquilEpsilon1/"
traj_path =  "EquilEpsilon1/InitialConditions/"
sigma = 1.0
epsilon = 1.0
constants = [sigma, epsilon]
dt = 0.00001
M = 1
sim_time = 0.05
equilibration_time = sim_time/2
Ncube = 5
N = Ncube ** 3
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
        force=force_lj,
        constants=constants,
        periodic=True,
        from_traj=traj_path,
        heat_bath=False,
        T_heat=None,
    )
trajs = [grid, pos, vels, E_pot]
save_trajectories(trajs, prefix=prefix)
E_kin = calculate_kinetic_energy(vel_trajectory = vels, mass=M)
E_tot = E_kin + E_pot 
E_diff = E_tot - E_tot[0]
axis_label=["Time $t\\cdot \\frac{\epsilon}{m\sigma^2}$", 
            "Energy differnce $\\frac{\delta E(t)}{\epsilon}$"]
plot_single(prefix_data = prefix, 
            out_path="E_diff",
            trajectory = E_diff, 
            grid = grid, 
            axis_label=axis_label)
