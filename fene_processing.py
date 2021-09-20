import numpy as np 

def radius_of_gyration(pos_traj, N): 
    """Calculates the radius of gyration for a given position trajectory

    Args:
        pos_traj (np.ndarray): [description]
        N (int): N_particles
    """
    mean_pos = np.mean(pos_traj, axis=0)
    diff = mean_pos - pos_traj
    r_g = (np.sum(diff, axis=0)/N)**(1/2)
    return r_g

def end_to_end_dist(pos_traj): 
    """Calculates the end-to-end distance of a given position trajectory
        (not the expecation value)
    Args:
        pos_traj ([type]): [description]
    """
    return np.abs(np.sum(pos_traj, axis=0)-pos_traj[0,:,:])

def isochoric_heat(u_tot, k, T):
    """

    :param u_tot:
    :return:
    """
    c_v = (np.mean(u_tot**2)-np.mean(u_tot)**2)/(k*T**2)
    return c_v

def rmsd(pos_traj): 
    """Calculates the rmsd of a position trajectory

    Args:
        pos_traj ([type]): [description]
    """
    raise NotImplementedError
