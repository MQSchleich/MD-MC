from scipy.signal import find_peaks
import numpy as np 
import matplotlib.pyplot as plt

#TODO implement method to find the distance https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
def get_distances(ener_traj, height): 
    ener = np.load(ener_traj)
    peaks, properties = find_peaks(ener, height=height)
    plt.plot(ener)
    plt.plot(peaks, ener[peaks], "x")
    plt.show()
    plt.close()
    diff = np.diff(peaks)
    print(peaks)
    print(ener[peaks[0]]-ener[peaks][1])
if __name__ =="__main__": 
    ener_traj = "EquilEpsilon1/e_pot.npy"
    gird_traj = "EquilEpsilon1/gid.npy"
    height = 100000 
    get_distances(ener_traj=ener_traj, height=height)

