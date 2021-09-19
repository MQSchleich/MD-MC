from lj_force import force_lj_fast, force_lj
from simulation import simulate
from post_simulation import save_trajectories
from data_processing import plot_k, plot_p, err_plot

prefix = "ErrEpsilon1/"
traj_path =  "ErrEpsilon1/InitialConditions/"
sigma = 1.0
epsilon = 1.0
constants = [sigma, epsilon]
dt = 0.00001
M = 1
sim_time = 0.005
equilibration_time = sim_time/2
Ncube = 5
N = Ncube ** 3
L = 8
T0 = 1.6
factors = [1]

for k in factors:
    print("Running simulation with k= ", str(k))
    prefix_new = prefix + "k_" + str(k)
    time_step = dt * k
    steps = int(sim_time / time_step)
    trajs = simulate(
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
    save_trajectories(trajs, prefix=prefix_new)
plot_k(prefix_data=prefix, mass=M, factors=factors)
plot_p(prefix_data=prefix, equilibration_time=equilibration_time, mass=M, factors = factors)
err_plot(prefix_data=prefix, delta_t=dt, mass=M, equilibration_time=equilibration_time, factors = factors)
