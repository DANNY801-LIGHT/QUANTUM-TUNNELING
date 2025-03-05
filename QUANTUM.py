import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0     # Mass of the particle
V0 = 5.0    # Height of the potential barrier
a = 1.0     # Width of the potential barrier
N = 1000    # Number of grid points
x_max = 5.0 # Maximum x value
dx = 2 * x_max / N  # Grid spacing
dt = 0.01   # Time step for animation

# Double barrier potential function
def V(x):
    return V0 * (np.where((x > -a) & (x < -a/2), 1, 0) + np.where((x > a/2) & (x < a), 1, 0))

# Discretized Schrödinger equation
def schrodinger_eq(psi, x):
    second_deriv = np.gradient(np.gradient(psi, dx), dx)
    return - (hbar**2 / (2 * m)) * second_deriv + V(x) * psi

# Initial wavefunction (Gaussian wave packet)
x = np.linspace(-x_max, x_max, N)
psi = np.exp(-x**2) * np.exp(1j * 2 * x)  # Gaussian with momentum

# Normalize the initial wavefunction
psi /= np.sqrt(np.trapz(np.abs(psi)**2, x))

# Time evolution of the wavefunction
def time_evolution(psi, dt):
    return psi + dt * (-1j) * schrodinger_eq(psi, x)

# Energy spectrum calculation
def energy_spectrum(psi, x):
    H_psi = - (hbar**2 / (2 * m)) * np.gradient(np.gradient(psi, dx), dx) + V(x) * psi
    return np.trapz(np.conj(psi) * H_psi, x)

# Animated plot setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.set_xlim(-x_max, x_max)
ax1.set_ylim(0, 1.5)
ax1.set_xlabel('Position (x)')
ax1.set_ylabel('Probability Density')
ax1.set_title('Time Evolution of Wavefunction')

ax2.set_xlim(-x_max, x_max)
ax2.set_ylim(0, V0 + 1)
ax2.set_xlabel('Position (x)')
ax2.set_ylabel('Potential / Energy')
ax2.set_title('Potential Barrier and Energy Spectrum')

# Initialize plots
probability_density_line, = ax1.plot([], [], label='Probability Density')
potential_line, = ax2.plot(x, V(x), label='Potential Barrier', linestyle='--')
energy_levels_line, = ax2.plot([], [], 'ro', label='Energy Levels')

# Animation function
def animate(frame):
    global psi
    psi = time_evolution(psi, dt)
    psi /= np.sqrt(np.trapz(np.abs(psi)**2, x))  # Normalize after each step

    # Update probability density plot
    probability_density = np.abs(psi)**2
    probability_density_line.set_data(x, probability_density)

    # Update energy spectrum
    energies = [energy_spectrum(psi, x)]
    energy_levels_line.set_data(x, energies)

    return probability_density_line, energy_levels_line

# Create animation
ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)

# Show plot
plt.tight_layout()
plt.legend()
plt.show()