import numpy as np 
import matplotlib.pyplot as plt 

def plot_inverse_fourier(ener_traj, grid, save_path): 
    ener = np.load(ener_traj)
    grid = np.load(grid)
    assert len(grid) == len(ener), "Len of grid is not equal to len of ener"
    fft = np.fft.fft(ener)
    #freq = np.fft.fftfreq(len(grid), d=grid[1]-grid[0])
    ifft = np.fft.ifft(fft.real)
    plt.plot(grid, ifft)
    plt.xlabel("time $t$ in $\\left(\\frac{\epsilon}{m\sigma^2}\\right)^{\\frac{1}{2}}$")
    plt.ylabel("Energy in $\epsilon$")
    #plt.ylim(-1.4*10**(5), +1.0*10**(5))
    #plt.xlim(-6*10**(3), +6*10**(3))
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    plt.plot(grid, ener, label="Simulation")
    plt.plot(grid, ifft.real, label="Inverse FFT")
    plt.xlabel("time $t$ in $ \\left(\\frac{\epsilon}{m\sigma^2}\\right)^{\\frac{1}{2}}$")
    plt.ylabel("Energy in $\epsilon$")
    plt.legend()
    plt.savefig(save_path+"overlay", dpi=300)
    plt.show()
    plt.close()


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
    #plt.ylim(-4*10**(6), +4*10**(6))
    #plt.xlim(-2*10**(7), +2*10**(7))
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__": 
    plot_fourier("ZeroKelvin025T/e_pot.npy", 
                 "ZeroKelvin025T/grid.npy", 
                 save_path="ZeroKelvin025T/fourier.png")
    plot_inverse_fourier("ZeroKelvin025T/e_pot.npy", 
                         "ZeroKelvin025T/grid.npy", 
                 save_path="ZeroKelvin025T/inverse_fourier")