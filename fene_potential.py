from operator import pos
import numpy as np
from numba import njit

@njit
def fene_chain_potential(positions, constants = [1,1]):
    """
    """
    r_max, K = constants
    num = positions.shape[0]
    potential = np.zeros((num, num))
    for i in range(0,num-1): 
        for j in range(i+1, i+2): 
            distance = np.linalg.norm(positions[i,:]- positions[j,:])
            potential[i, j] = -0.5*K*r_max**2*np.log(1-(distance/r_max)**2)

    #potential[np.tril_indices_from(potential)] -= potential[np.triu_indices_from(potential)]
    return np.sum(potential)


def fene_ring_potential(positions, r_max, K): 
    """Calculates the FENE potential on a ring 

    Args:
        positions ([type]): [description]
        r_max ([type]): [description]
        K ([type]): [description]
    """
    raise NotImplementedError

if __name__ == "__main__": 
    from initialization import InitFeneChain
    k = 1
    r_max = 1 
    N = 48
    L = (N*r_max)/3
    constants = [r_max, k]
    positions = InitFeneChain(48, L, constants,) 
    print(fene_chain_potential(positions, constants))