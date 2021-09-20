import numpy as np
import csv


def compute_statistics_c(ener_traj, pressure_traj, save_path="Exam1c/"):
    mean_ener = np.mean(ener_traj)
    mean_pressure = np.mean(pressure_traj)
    err_ener = np.std(ener_traj)
    rel_err_ener = err_ener / mean_ener
    err_pressure = np.std(pressure_traj)
    rel_err_pressure = err_pressure / mean_pressure
    d = {
        "avg_e_pot": mean_ener,
        "std_e_pot": err_ener,
        "rel_err_epot": rel_err_ener,
        "avg_pressure": mean_pressure,
        "std_pressure": err_pressure,
        "rel_err_pressure": rel_err_pressure,
    }

    save_dict(d, save_path=save_path)


def compute_statistics_b(ener_traj, momentum_traj, save_path="Exam1b/"):

    mean_de = np.mean(ener_traj - ener_traj[0])
    E = np.mean(ener_traj)
    rmsd = (mean_de ** 2 / E ** 2) ** (1 / 2)
    abs_dev_p = (np.mean(momentum_traj - momentum_traj[0]) ** 2) ** (1 / 2)
    save_statistics_b([rmsd, abs_dev_p], save_path)


def save_statistics_b(vals, save_path):

    d = {"rmsd": vals[0], "abs_dev_p": vals[1]}
    w = csv.writer(open(save_path + "stats.csv", "w"))
    for key, val in d.items():
        w.writerow([key, val])


def save_dict(d, save_path):
    w = csv.writer(open(save_path + "stats.csv", "w"))
    for key, val in d.items():
        w.writerow([key, val])
        print("Saved the date in " + save_path)
