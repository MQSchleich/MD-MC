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
            distance_sq = np.sum((positions[i,:]- positions[j,:])**2)
            potential[i, j] = -0.5*K*distance_sq
    return np.sum(potential)



@njit
def harmonic_potential_periodic(positions, constants): 
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
            distance_sq = difference_x**2+difference_y**2+difference_z**2
            potential[i, j] = -0.5*K*distance_sq
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
    distance_sq = difference_x**2+difference_y**2+difference_z**2       
    potential[1, -1] = -0.5*K*distance_sq
    return np.sum(potential)

