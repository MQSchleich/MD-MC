import numpy as np
from numpy.lib.function_base import _diff_dispatcher 

def fene_chain_force(positions, constants=[1,1]):
    """Calculates the fene force for a chain

    Args:
        positions ([type]): [description]
        constants ([type]): [description]
        box_length ([type]): [description]

    Returns:
        [type]: [description]
    """
    r_max, K = constants
    num = positions.shape[0]
    force = np.zeros((num, num, 3))
    for i in range(0,num-1): 
        for j in range(i+1, i+2): 
            difference=positions[i,:]- positions[j,:]
            distance = np.linalg.norm(difference)
            force[i, j, :] = (-K*r_max**2*distance)/(distance**2-r_max**2)*(difference/distance)
            force[j, i, :] -= force[i, j, :]
   
    return np.sum(force, axis=1))


def force_fene_ring(positions, constants, box_length):
    """Calculates the fene force for a chain

    Args:
        positions ([type]): [description]
        constants ([type]): [description]
        box_length ([type]): [description]

    Returns:
        [type]: [description]
    """
    raise NotImplementedError

if __name__ == "__main__": 
    positions = np.random.random((5,3))
    fene_chain_force(positions)