import numpy as np
import csv
from fene_processing import isochoric_heat
def compute_statistics_fc(ener_traj, r_g_traj, r_e_traj, constants, save_path="Exam2a/"): 
    """computes for statistcs for the exercise 2a

    Args:
        pos_traj ([type]): [description]
        save_path (str, optional): [description]. Defaults to "Exam1c/".
    """
    k_b, T = constants
    re_mean = np.mean(r_e_traj)
    re_std = np.std(r_e_traj)
    re_rel = re_std/re_mean

    rg_mean = np.mean(r_g_traj)
    rg_std = np.std(r_g_traj)
    rg_rel = rg_std/rg_mean

    c_v = isochoric_heat(ener_traj, k=k_b, T=T)
    cv_rel = 2*np.std(ener_traj)/np.mean(ener_traj)
    std_cv = c_v*cv_rel

    d = {
        "re_mean": re_mean,
        "re_std": re_std, 
        "re_rel": re_rel, 

        "rg_mean": rg_mean, 
        "rg_std": rg_std, 
        "rg_rel": rg_rel, 

        "c_v": c_v, 
        "std_cv": std_cv,
        "cv_rel": cv_rel
    }
    save_dict(d, save_path=save_path)



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
