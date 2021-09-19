from sys import prefix
import matplotlib.pyplot as plt
import numpy as np

from helpers import linear_regression

from potentials import energy_lj_fast


def plot_p(
    prefix_data, out_path=None, 
    delta_t = 0.002,
    equilibration_time=0.2, 
    mass=1,
    factors=None
):
    """

    :return:
    """
    if factors is None: 
        factors = [1, 2, 4]
    if out_path is None:
        out_path = prefix_data + "k_plot"
    save_path = out_path + "p_"
    path = prefix_data + "k_"
    for i in [1, 2, 3, 4]:
        new_path = save_path + str(i)
        for k in factors:
            load_path = path + str(k)
            grid = np.load(load_path + "grid.npy")[int(equilibration_time/(delta_t*k)):]
            total_energy = get_total_energy(path=load_path, mass=mass)[int(equilibration_time/(delta_t*k)):]
            energy_diff = total_energy - total_energy[0]

            plt.plot(grid, grid ** (-i) * energy_diff, label="k = " + str(k))
            plt.xlabel("Time $t$ in $ \\left(\\frac{\epsilon}{m\sigma^2}\\right)^{\\frac{1}{2}}$")
            plt.legend()
            plt.ylabel("Energy differnce $\\delta E(t)$ in $\\epsilon$")
            plt.xticks()
            plt.legend()
            plt.yticks()
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.savefig(new_path, dpi=300)
        plt.close()


def zoom_plot(
    path="NewExamData/Ex1a/",
    save_path="NewExamPlots/Exam_Energy_diff_factor_zoom",
    delta_t=0.002,
    start_traj=2,
    mass=1,
):
    for i in [2]:
        for k in [1, 2, 4]:
            grid = np.load(path + "Ex1a_grid_" + str(k) + ".npy")
            energy_diff = np.load(path + "Ex1a_energy_diff_" + str(k) + ".npy")
            energy_diff = energy_diff[int(start_traj / (delta_t * k)) :]
            grid = grid[int(start_traj / (delta_t * k)) :]
            plt.plot(grid, grid ** (-i) * energy_diff, label="k = " + str(k))
            plt.xlabel("time [$t$]")
            plt.ylabel("Error term [$\delta E(t)(\Delta t)^{-p}/\epsilon$]")
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.legend()
            plt.xticks()
            plt.yticks()
            plt.legend()
        plt.savefig(
            save_path + "Exam_Energy_diff_factor_zoom" + str(i) + "vst" + ".png"
        )
        plt.close()


def err_plot(prefix_data, out_path=None, delta_t=0.002, mass=1, factors=None, equilibration_time=1):
    """

    :return:
    """
    if out_path is None:
        out_path = prefix_data + "err_plot"
    path = prefix_data + "k_"
    if factors is None:
        factors = [1, 2, 4]
    err_list = []
    dt_list = []
    for k in factors:
        load_path = path+str(k)
        total_energy = get_total_energy(path=load_path, mass=mass)
        energy_diff = total_energy - total_energy[0]
        interesting_diff = energy_diff[int(equilibration_time / (delta_t*k)) :]
        mean_err = np.linalg.norm(interesting_diff, ord=1)
        dt_list.append(delta_t ** k)
        err_list.append(mean_err)
        plt.xlabel("$\\log(|\\Delta t\\cdot k|_{\infty})$")
        plt.ylabel(" $\log(\\delta E(t))$")
        plt.xticks()
        plt.yticks()
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    x_val = np.log(np.array(dt_list))
    y_val = np.log(np.array(err_list))
    fitted_line, slope, shift, pearson_r = linear_regression(
        x_val, y_val)
    plt.scatter(
        np.log(np.array(dt_list)),
        np.log(np.array(err_list)),
        label="$\log(\delta E(t))$",
        c="orange"
    )
    plt.plot(
        np.log(np.array(dt_list)),
        fitted_line,
        label="Linear fit, r=%.4f" % pearson_r + ", slope = %.2f" % slope,
    )
    plt.legend()
    plt.savefig(out_path + str(k) + ".png", dpi=300)
    plt.close()


def plot_k(prefix_data, out_path=None, mass=1, factors=None):
    """Total Energy plot

    Args:
        prefix_data ([type]): folder to saved trajectories
        out_path ([type]): where to save plots
    """
    if factors is None: 
        factors = [1, 2, 4] 
    if out_path is None:
        out_path = prefix_data + "k_plot"
    path = prefix_data + "k_"
    for k in factors:
        load_path = path + str(k)
        grid = np.load(load_path + "grid.npy")
        total_energy = get_total_energy(path=load_path)
        energy_diff = total_energy - total_energy[0]
        plt.plot(grid, energy_diff, label="k = " + str(k))
        plt.xlabel("Time $t$ in $ \\left(\\frac{\epsilon}{m\sigma^2}\\right)^{\\frac{1}{2}}$")
        plt.legend()
        plt.ylabel("Energy difference $\\delta E(t)$ in $\\epsilon$")
        plt.xticks()
        plt.yticks()
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_single(prefix_data, trajectory, grid, out_path=None, axis_label=["", ""]):
    """

    :param prefix:
    :param trajectory:
    :param grid:
    :param axis_label:
    :return:
    """
    if out_path is None:
        out_path = prefix_data + "k_plot"
    path = prefix_data + out_path
    plt.plot(grid, trajectory)
    plt.xticks()
    plt.yticks()
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.savefig(path, dpi = 300)
    plt.close()


def plot_components(prefix_data, trajectory, grid, axis_label=["", ""], out_path=None):
    """
    plots the cartesian components of a trajectory
    :param prefix: string containing the filepath
    :param trajectory: trajectory of components
    :return:
    """
    if out_path is None:
        out_path = prefix_data 
    path = prefix_data + out_path
    energy_diff_x, energy_diff_y, energy_diff_z = trajectory
    energy_diff_x = energy_diff_x - energy_diff_x[0]
    energy_diff_x = energy_diff_y - energy_diff_y[0]
    energy_diff_x = energy_diff_z - energy_diff_z[0]
    plt.plot(grid, energy_diff_x, label="x component")
    plt.plot(grid, energy_diff_y, label="y component")
    plt.plot(grid, energy_diff_z, label="z component")
    plt.xticks()
    plt.yticks()
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.savefig(out_path + "total_mom.png")
    plt.close()


def energy_diff(cartesian_trajectory):
    """
    Calculates the difference energy along the cartesian coordinates
    :param cartesian_trajectory:
    :return:
    """
    e_x, e_y, e_z = cartesian_trajectory
    e_diff_x = e_x - e_x[0]
    e_diff_y = e_y - e_y[0]
    e_diff_z = e_z - e_z[0]
    return [e_diff_x, e_diff_y, e_diff_z]


def calculate_energy(positions, vel_trajectory, mass):
    """
    calculates the energies
    :param positions:
    :param vel_trajectory:
    :param mass:
    :return:
    """
    energy = get_total_energy(
        positions=positions, vel_trajectory=vel_trajectory, mass=mass
    )
    diff_energy = energy - energy[0]
    return (energy, diff_energy)


def calculate_kinetic_energy(vel_trajectory, mass):
    """
    Calculates the kinetic energy
    :param vel_trajectory:
    :param mass:
    :return:
    """
    vels = np.linalg.norm(vel_trajectory, axis=1)
    e_kin = 0.5 * mass * vels ** 2
    e_kin = np.sum(e_kin, axis=0)
    return e_kin


def get_total_energy(path, mass=1):
    """calculates the energies from the trajectories

    Args:
        path (str): [description]
        positions (np.ndarr): [description]
        mass (np.ndarr): [description]

    Returns:
        [type]: total energy
    """
    e_pot = np.load(path + "e_pot.npy")
    vel_trajectory = np.load(path + "vel.npy")
    e_kin = calculate_kinetic_energy(vel_trajectory, mass)
    return e_kin + e_pot


def load_trajectory():
    """
    loads vels and positions from trajectory
    :return:
    """
    pass
