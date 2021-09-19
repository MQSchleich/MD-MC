import numpy as np
import matplotlib.pyplot as plt


def replot_ener_diff(traj_path, grid_path, save_path):
    """[summary]

    Args:
        traj_path ([type]): [description]
        grid_path ([type]): [description]
    """

    axis_label = [
        "Time $t$ in $\\frac{\epsilon}{m\sigma^2}$",
        "Energy difference $\\delta E(t)$ in $\\epsilon$",
    ]
    grid = np.load(grid_path)
    ener = np.load(traj_path)
    ener = ener - ener[0]
    plt.plot(grid, ener)
    plt.xticks()
    plt.yticks()
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.savefig(save_path, dpi=300)
    print("saved figure.")


if __name__ == "__main__":
    replot_ener_diff(
        "EquilEpsilon1/e_pot.npy",
        "EquilEpsilon1/grid.npy",
        "EquilEpsilon1/eps1_EquilSample.png",
    )
