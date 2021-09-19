from lj_force import force_lj, force_lj_fast
from simulation import simulate
from post_simulation import save_trajectories
from data_processing import plot_k, plot_p, err_plot
from initialization import InitMonteCarlo, InitVelocity
import numpy as np 

prefix = "Exam1a/"
traj_path = None  # "InitialConditions/"
sigma = 1.0
sigma = 0.25
epsilon = 0.1
constants = [sigma, epsilon]
dt = 0.0001
M = 1
sim_time = 0.8
equilibration_time = sim_time/2
Ncube = 128
#N = Ncube ** 3

L = 8
T0 = 1.6
factors = [1, 2, 4]
init_pos = InitMonteCarlo(N=Ncube, L=L, constants=constants)
init_vels =  InitVelocity(N=Ncube, T0=T0, mass=M)
np.save(arr= init_pos, file=prefix+"pos.npy")
np.save(arr= init_vels, file=prefix+"vel.npy")
from_traj = prefix
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
        force=force_lj_fast,
        constants=constants,
        periodic=True,
        from_traj=from_traj,
        heat_bath=False,
        T_heat=None,
    )
    assert trajs[1].shape[0] == 128, "Wrong trajecotry "+str(trajs[1].shape)
    save_trajectories(trajs, prefix=prefix_new)
plot_k(prefix_data=prefix, mass=M)
plot_p(prefix_data=prefix, equilibration_time=equilibration_time, mass=M)
err_plot(prefix_data=prefix, delta_t=dt, mass=M, equilibration_time=equilibration_time)


