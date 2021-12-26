import numpy as np


def zero_point(config_path, save_path):
    pos = np.load(config_path + "pos.npy")
    vel = np.load(config_path + "vel.npy")
    vel[:, :, -1] = 0
    np.save(arr=pos[:, :, -1], file=save_path + "pos.npy")
    np.save(arr=vel[:, :, -1], file=save_path + "vel.npy")


if __name__ == "__main__":
    config_path = "Equilepsilon1long/"
    save_path = "ZeroKelvin/InitialConditions/"
    zero_point(config_path=config_path, save_path=save_path)
