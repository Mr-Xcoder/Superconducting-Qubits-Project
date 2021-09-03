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

for eigenvalue, eigenvector in zip(*np.linalg.eigh(phi)):
    cos_phi += np.cos(2*np.pi*eigenvalue/phi0-2*np.pi*phi_ext_ratio)*np.outer(eigenvector, eigenvector)  # am verificat ca sumele de outer dau 1

H = Hharmonic+Ej*(np.identity(N)-cos_phi)
harmonic_eigenvalues = np.linalg.eigh(Hharmonic)[0]
harmonic_eigenvectors = np.linalg.eigh(Hharmonic)[1]
anharmonic_eigenvalues = np.linalg.eigh(H)[0]
anharmonic_eigenvectors = np.linalg.eigh(H)[1]


# TODO: draw y=Ek lines instead of scattering points
def plot_energies():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid()
    plt.scatter(2*np.pi*fluxes/phi0, anharmonic_eigenvalues, label="QAHO Energy", marker=".")
    plt.scatter(2*np.pi*fluxes/phi0, harmonic_eigenvalues, label="QHO Energy", marker=".")
    plt.xlabel(r"Phase $2\pi\dfrac{\Phi}{\Phi_0}$")
    plt.ylabel("Oscillator energy")
    ax.legend()
    plt.show()


plot_energies()


def diff_plot():
    print(np.diff(harmonic_eigenvalues))
    print(np.diff(anharmonic_eigenvalues))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(*zip(*enumerate(np.diff(anharmonic_eigenvalues))), label="QAHO", marker=".")
    plt.scatter(*zip(*enumerate(np.diff(harmonic_eigenvalues))), label="QHO", marker=".")
    plt.xlabel("Level number")
    plt.ylabel("Differences between consecutive energy levels")
    ax.legend()
    plt.show()


diff_plot()


def debug():
    print(cos_phi)
    print("harmonic", Hharmonic)
    print("josephson", Ej * cos_phi)
    print("h", Hharmonic - Ej * cos_phi)
    print(np.linalg.eigh(H)[0])
    print(np.linalg.eigh(H)[0] - np.linalg.eigh(Hharmonic)[0])


# debug()

def V(phi):
    return phi**2*phi0**2/(2*L)+(1-np.cos(2*np.pi*phi-2*np.pi*phi_ext_ratio))*Ej


def plot_potential():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid()
    x = np.arange(-1, 1, 0.01)
    plt.plot(x, V(x), label=r"$V(\phi)$")
    plt.xlabel(r"$\phi\equiv 2\pi\left(\dfrac{\Phi-\Phi_{ext}}{\Phi_0}\right)$")
    plt.ylabel(r"$V(\phi)$")
    ax.legend()
    plt.show()


plot_potential()


def psi_form(k, phi, phi_zpf):
    return np.exp(-phi**2/(4*phi_zpf**2)) * scipy.special.eval_hermite(int(k), phi/(phi_zpf*np.sqrt(2)))


def psi(k, phi, phi_zpf):
    return np.sqrt(phi0/(2**k*math.factorial(k)))*((np.pi*hbar*np.sqrt(L/C))**(-1/4))*psi_form(k, phi, phi_zpf)


def psi2(k, phi, phi_zpf):
    return sum(psi(i, phi, phi_zpf)*np.dot(harmonic_eigenvectors[i], anharmonic_eigenvectors[k]) for i in range(N))

#    print(scipy.integrate.quad(lambda phi: (psi_form(k, phi, phi_zpf))**2, -np.inf, np.inf))
#    return psi_form(k, phi, phi_zpf) * np.sqrt(1/scipy.integrate.quad(lambda phi: (psi_form(k, phi, phi_zpf))**2, -np.inf, np.inf)[0])
# TODO: de ce scipy greseste normarea?

def plot_wavefunctions():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid()
    for k in range(6):
        phi_zpf = 1
        n = 5
        x = np.arange(-n * phi_zpf, n * phi_zpf + phi0, phi_zpf / 20)
        plt.plot(x, [psi(k, j, phi_zpf) ** 2 + 2 * k for j in x], label=fr"$|\psi_{k}(\phi)|^2$ (th.)")
        plt.plot(x, [psi2(k, j, phi_zpf) ** 2 + 2 * k for j in x], label=fr"$|\psi_{k}(\phi)|^2$ (exp.)")
        plt.xlabel(r"$\varphi=\dfrac{\phi}{\phi^{ZPF}}$")
        plt.ylabel(r"$|\Psi|^2$")
    ax.legend(loc=2, prop={'size': 7})
    plt.show()

plot_wavefunctions()

#evolution_op = np.zeros((N, N), float)
"""
def make_animation():
    fig = plt.figure()
    ax = plt.axes()
    line, = ax.plot([], [], lw=2)
    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        phi_zpf = 1
        n = 5
        x = np.arange(-n * phi_zpf, n * phi_zpf + phi0, phi_zpf / 20)
        y = np.exp()
        line.set_data(x, y)
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=200, interval=20, blit=True)

    plt.show()
"""

# H = Q^2/2C+Phi^2/2L+Ej(1-cos(phi)); phi = 2pi phi_j/phi_0

