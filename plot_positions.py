import numpy as np
import matplotlib.pyplot as plt


def plot_particle(traj_path, num=1):

    traj = np.load(traj_path + "pos.npy")[num, :, :]
    plt.plot(traj[0, :])
    plt.plot(traj[1, :])
    plt.plot(traj[2, :])
    plt.show()


if __name__ == "__main__":
    plot_particle("Exam1a/k_4")
