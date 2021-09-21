import numpy as np 
import matplotlib.pyplot as plt 

def plot_fourier(ener_traj, grid, save_path):
    """Plots a fourier series for a given energy trajectory

    Args:
        ener_traj ([type]): [description]
    """
    ener = np.load(ener_traj)
    grid = np.load(grid)
    assert len(grid) == len(ener), "Len of grid is not equal to len of ener"
    fft = np.fft.fft(ener)
    freq = np.fft.fftfreq(len(grid), d=grid[1]-grid[0])
    plt.plot(freq/(grid[1]-grid[0]), fft.real)
    plt.xlabel("Frequency $\\nu$ in $\\frac{1}{\\tau}$")
    plt.ylabel("Intensity in a.u.")
    plt.ylim(-1.4*10**(5), +1.0*10**(5))
    plt.xlim(-6*10**(3), +6*10**(3))
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__": 
    plot_fourier("Exam2a/e_pot.npy", 
                 "Exam2a/grid.npy", 
                 save_path="Exam2a/fourier.png")