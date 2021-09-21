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
    return np.sum(potential)

@njit
def fene_ring_potential(positions, constants=[1,15]): 
    """Calculates the FENE potential on a ring 

    Args:
        positions ([type]): [description]
        r_max ([type]): [description]
        K ([type]): [description]
    """
    r_max, K = constants
    Ncube = positions.shape[0]
    box_length= Ncube*r_max/3
    num = positions.shape[0]
    potential = np.zeros((num, num))
    for i in range(0,num-1): 
        for j in range(i+1, i+2): 
            difference = positions[j,:]- positions[i,:]
            difference_x=(difference[0])
            difference_y=(difference[1])
            difference_z=(difference[2])
            if difference_x > box_length/2: 
                difference_x -= box_length/2
            elif difference_x<= -box_length/2: 
                difference_x += box_length/2
            if difference_y > box_length/2: 
                difference_y -= box_length/2
            elif difference_y <= -box_length/2: 
                difference_y += box_length/2
            if difference_z > box_length/2: 
                difference_z -= box_length/2
            elif difference_z <= -box_length/2: 
                difference_z += box_length/2
            distance = np.sqrt(difference_x**2+difference_y**2+difference_z**2)
            potential[i, j] = -0.5*K*r_max**2*np.log(1-(distance/r_max)**2)
    difference = positions[j,:]- positions[i,:]
    difference_x=(difference[0])
    difference_y=(difference[1])
    difference_z=(difference[2])
    if difference_x > box_length/2: 
        difference_x -= box_length/2
    elif difference_x<= -box_length/2: 
        difference_x += box_length/2
    if difference_y > box_length/2: 
        difference_y -= box_length/2
    elif difference_y <= -box_length/2: 
        difference_y += box_length/2
    if difference_z > box_length/2: 
        difference_z -= box_length/2
    elif difference_z <= -box_length/2: 
        difference_z += box_length/2
    distance = np.sqrt(difference_x**2+difference_y**2+difference_z**2)
            
    potential[1, -1] = -0.5*K*r_max**2*np.log(1-(distance/r_max)**2)
    return np.sum(potential)


if __name__ == "__main__": 
    from initialization import InitFeneChain
    k = 1
    r_max = 1 
    N = 48
    L = (N*r_max)/3
    constants = [r_max, k]
    positions = InitFeneChain(4, L, constants,) 
    print(fene_ring_potential(positions, constants))