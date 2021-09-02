import numpy as np
from scipy.constants import *
import scipy.special
import scipy.integrate
import matplotlib.pyplot as plt
import math
from matplotlib import animation


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
phi_ext_ratio = 0.25       # Phi_ext / Phi0
w0 = 1/np.sqrt(L*C)


def E(n):
    return hbar*w0*(n+1/2)


def psi_form(k, phi, phi_zpf):
    return np.exp(-phi**2/(4*phi_zpf**2)) * scipy.special.eval_hermite(int(k), phi/(phi_zpf*np.sqrt(2)))


def psi(k, phi, phi_zpf):
    return np.sqrt(phi0/(2**k*math.factorial(k)))*((np.pi*hbar*np.sqrt(L/C))**(-1/4))*psi_form(k, phi, phi_zpf)


def plot_wavefunctions():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid()
    for k in range(6):
        phi_zpf = 1
        n = 7
        x = np.arange(-n * phi_zpf, n * phi_zpf + phi0, phi_zpf / 20)
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(x, [psi(k, j, phi_zpf) + E(k)*10**24 for j in x], label=fr"$\psi_{k}(\Phi)$", color=color)
        plt.xlabel(r"$\Phi$")
        plt.ylabel(r"$E_n$ și $\Psi_n(\Phi)$")
        plt.title(r"Energiile $E_n$ și funcțiile $\psi_n$ asociate lor - oscilator LC armonic")
        plt.plot(x, [E(k)*10**24]*len(x), "--", label=fr"$E_{k}$", color=color)
    ax.legend(loc=2, prop={'size': 7})
    plt.xlim([-8.5, 8.5])
    plt.ylim([0, 14])
    plt.show()

plot_wavefunctions()
