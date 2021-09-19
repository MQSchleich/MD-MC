import numpy as np
import warnings

def load_initials(prefix): 
    """Load initial conditions from another trajectory

    Args:
        prefix ([type]): [description]

    Returns:
        [type]: [description]
    """
    pos = load_traj(prefix+"pos.npy")
    vels = load_traj(prefix+"vel.npy")
    return [pos, vels]



def load_traj(path): 
    """path to trajectory

    Args:
        path (string): string
    """
    pos = np.load(path)
    try:
        return pos[:,:, -1]
    except:
        warnings.warn("Loading an initial configuation not a trajectory")
        return pos

