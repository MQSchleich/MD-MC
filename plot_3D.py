import matplotlib.pyplot as plt
import numpy as np


def plot_positions(path_traj, single=False):
    """Plots a 3D viz of a given sample.

    Args:
        path_traj ([type]): [description]
        save_path ([type]): [description]
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    positions = np.load(path_traj)

    if single:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], marker="o")
    else:
        ax.scatter(
            positions[:, 0, -1], positions[:, 1, -1], positions[:, 2, -1], marker="o"
        )
    ax.set_xlabel("X coordinate $\\frac{l_x}{\\sigma}$")
    ax.set_ylabel("Y coordinate $\\frac{l_y}{\\sigma}$")
    ax.set_zlabel("Z coordinate $\\frac{l_z}{\\sigma}$")

    plt.show()


if __name__ == "__main__":
    # plot_positions("Exam1a/pos.npy",single = True)
    plot_positions("EquilEpsilon1/pos.npy")
    # plot_positions("ExamErrOrder/k_1pos.npy")
