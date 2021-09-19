from lj_force import force_lj
from simulation import simulate
from post_simulation import save_trajectories
from data_processing import plot_k, plot_p, err_plot

prefix = "ExamErrOrder/"
traj_path = None  # "InitialConditions/"
sigma = 1.0
epsilon = 1.0
constants = [sigma, epsilon]
dt = 0.0001
M = 1
sim_time = 20
equilibration_time = 10
Ncube = 5
N = Ncube ** 3
L = 8
T0 = 1.6
factors = [10, 100, 1000]

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
        from_traj=traj_path,
        heat_bath=False,
        T_heat=None,
    )
    save_trajectories(trajs, prefix=prefix_new)
plot_k(prefix_data=prefix, mass=M)
plot_p(prefix_data=prefix, equilibration_time=equilibration_time, mass=M)
err_plot(prefix_data=prefix, delta_t=dt, mass=M, equilibration_time=equilibration_time)
