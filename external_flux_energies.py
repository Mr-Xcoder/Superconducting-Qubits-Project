import numpy as np
from scipy.constants import *
import scipy.special
import scipy.integrate
import matplotlib.pyplot as plt
import math


L = 386*10**(-9)
C = 5.3*10**(-15)
Ec = e**2/(2*C)
Qzpf = np.sqrt(hbar/(2*np.sqrt(L/C)))
qzpf = Qzpf / (2*e)
Ej = 6.2 * 10**9 * hbar  # investigheaza comportamentul pentru * 10 ** (>14.5)
N = 150
phi_max = 10**(-11)
delta_phi = 2*phi_max/(N-1)
phi0 = h/(2*e)
phi_zpf = np.sqrt(hbar*np.sqrt(L/C)/2)
omega0 = 1 / np.sqrt(L*C)
E = hbar*omega0*np.arange(1/2, N+1/2, 1)
fluxes = np.arange(-phi_max, phi_max+delta_phi/2, delta_phi)
Hharmonic= np.diag(E)
Dnmm1 = np.zeros((N, N), float)
Dnmp1 = np.zeros((N, N), float)

for m in range(N):
    for n in range(N):
        if m-1 == n:
            Dnmm1[m][n] = np.sqrt(m)
        if m+1 == n:
            Dnmp1[m][n] = np.sqrt(m+1)

phi = (Dnmm1 + Dnmp1) * phi_zpf
cos_phi = np.zeros((N, N), float)


def system_states(phi_ext_ratio):
    global cos_phi
    for eigenvalue, eigenvector in zip(*np.linalg.eigh(phi)):
        cos_phi += np.cos(2 * np.pi * eigenvalue / phi0 - 2 * np.pi * phi_ext_ratio) * np.outer(eigenvector, eigenvector)
    H = Hharmonic + Ej * (np.identity(N) - cos_phi)
    return np.linalg.eigh(H)


def plot_energies():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid()
    phi_ext_list = np.arange(0, 6, 0.1)
    for i in range(1, 6):
        plt.scatter(phi_ext_list, [system_states(x)[0][i] for x in phi_ext_list], label=r"$E_{%s}(\Phi_{ext})$"%str(i), marker=".")
    plt.xlabel(r"External flux ($\dfrac{\Phi_{ext}}{\Phi_0}$)")
    plt.ylabel("Oscillator energy levels")
    ax.legend()
    plt.show()

plot_energies()
