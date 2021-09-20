from operator import pos
import numpy as np

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

    potential[np.tril_indices_from(potential)] -= potential[np.triu_indices_from(potential)]
    print(potential)
    print(np.sum(potential, axis =1))


def fene_ring_potential(positions, r_max, K): 
    """Calculates the FENE potential on a ring 

    Args:
        positions ([type]): [description]
        r_max ([type]): [description]
        K ([type]): [description]
    """
    raise NotImplementedError

if __name__ == "__main__": 
    positions = np.random.random((5,3))
    fene_chain_potential(positions)