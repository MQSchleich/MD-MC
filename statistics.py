import numpy as np 
import csv


def compute_statistics_b(ener_traj, momentum_traj, save_path="Exam1b/"): 

    mean_de = np.mean(ener_traj-ener_traj[0])
    E = np.mean(ener_traj)
    rmsd = (mean_de**2 / E**2)**(1/2)
    abs_dev_p = (np.mean(momentum_traj-momentum_traj[0])**2)**(1/2)
    save_statistics_b([rmsd, abs_dev_p], save_path)


def save_statistics_b(vals, save_path): 

    d = {"rmsd": vals[0] ,
         "abs_dev_p": vals[1]}
    w = csv.writer(open(save_path+"stats.csv", "w"))
    for key, val in d.items():
        w.writerow([key, val])