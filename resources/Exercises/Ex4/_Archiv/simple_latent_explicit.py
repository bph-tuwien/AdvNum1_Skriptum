import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# -----------------------------
#   Helper Functions
# -----------------------------

#
#	Thermophysical functions
#
# vapour pressure at saturation
def fc_pvsat(T):
    return np.power(10, 7.625 * T / (241 + T) + 2.7877)


# vapour diffusivity depending on water content
def fc_deltav(w):
    a = 1.1 * 1e-7
    b = -1.57 * 1e-7
    return a + b * w


# thermal conductivity depending on water content
def fc_lambda(w):
    a = 0.23
    b = 6
    return a + b * w


# sorption curve
def fc_w_phi(phi):
    a = 700  # 1000
    b = 250 * 1e-6  # 146*1e-6
    c = 2.9  # 1.59
    res = a / np.power(1 - np.log(phi / 100) / b, 1 / c)
    return res * 1e-3


# sortpion curve the other way around
def fc_phi_w(w):
    a = 700  # 1000
    b = 250 * 1e-6  # 146*1e-6
    c = 2.9  # 1.59
    phi = np.zeros(len(w))
    phi = np.exp(b * (1 - np.power((a / (1000 * w)), c))) * 100
    return phi


# update local Fourier
def update_Fo(w_vec, rho, Cp, dx):
    k = fc_lambda(w_vec)  # variable properties
    Cp_vec = Cp + w_vec / rho
    # Fo_vec = k / rho / Cp_vec * dt / dx ** 2
    Fo_vec = k / rho / Cp_vec / dx ** 2  #  dt not needed because we solve ivp
    return Fo_vec


# update local mass Fourier
def update_Fow(w_vec, dx):
    deltav = fc_deltav(w_vec)  # variable properties
    # Fow = deltav * dt / dx ** 2
    Fow = deltav / dx ** 2     #  dt not needed because we solve ivp
    return Fow


# update local Fourier
def update_Cm(w, phi, T):
    epsilon = 0.001 * np.min(w)
    wp = w + epsilon
    phip = fc_phi_w(wp)
    dw = abs(wp - w)
    dphi = abs(phip - phi) / 100
    pvs = fc_pvsat(T)
    Cm = dw / dphi / pvs
    return Cm


# -----------------------------
# Input
# -----------------------------

sim_time = 3600 * 2
dx = 0.01

n = 100 + 2


# material properties
rho = 1000
Cp = 1000
k = 0.6
deltav = 1e-7

# Fields and initial conditions
T = np.ones(n) * 20
phi = np.ones(n) * 50
phi[0] = 60
phi[-1] = 60
w = fc_w_phi(phi)
pv = phi / 100 * fc_pvsat(T)



# Systemmatrix
K = np.eye(n, n, k=-1) * 1 + np.eye(n, n) * -2 + np.eye(n, n, k=1) * 1
K[0, 0], K[0, 1], K[-1, -1], K[-1, -2] = 0, 0, 0, 0


# Equation system
def coupled_equation_system(t, T_w):
    n = int(len(T_w) / 2)  # half index
    T, w = T_w[0:n], T_w[n:]
    Lv = 2400 * 1e3  # J/kg
    phi = fc_phi_w(w)
    pv = phi / 100 * fc_pvsat(T)
    Fo = update_Fo(w, rho, Cp, dx)
    Fow = update_Fow(w, dx)
    dTdt = Fo * np.dot(K, T) + Lv * Fow / (rho * Cp) * np.dot(K, pv)
    dTdt[0] = dTdt[-1] = 0
    dwdt = (Fow * np.dot(K, pv))
    dwdt[0] = dwdt[-1] = 0

    return np.hstack([dTdt, dwdt])
# -----------------------------
#   Simulation
# -----------------------------



T_w_0 = np.hstack([T, w])

t0 = 0  # Start time in seconds
tf = 100  # End time in seconds

t_eval = np.linspace(t0, tf, 10)

# Solve
print("Solving...")
sol = solve_ivp(coupled_equation_system, (t0, tf), T_w_0, t_eval=t_eval, dense_output=False, atol=1e-7, rtol=1e-5, method='Radau')
# sol = solve_ivp(coupled_equation_system, (t0, tf), T_w_0, t_eval=t_eval, dense_output=False, atol=1e-7, rtol=1e-5)
T_res = sol.y[0:n]
w_res = sol.y[n:]
phi_res = fc_phi_w(w_res)
pvap_res = phi_res / 100 * fc_pvsat(T_res)
print("Done solving.")

fig, ax = plt.subplots(2, 2)
x = np.linspace(0, n-2, 100)    # Only for material domain
for idx, value in np.ndenumerate(t_eval):
    ax[0, 0].plot(x, T_res[1:-1, idx[0]], label=f'{value:.2f} seconds')
ax[0, 0].set_title("Temperature")
ax[0, 0].set_ylabel("T in °C")
ax[0, 0].set_xlabel("x in m")
ax[0, 0].grid()

ax[0, 1].plot(x, phi_res[1:-1, :])
ax[0, 1].set_title("Realtive Humidity")
ax[0, 1].set_ylabel("phi in %")
ax[0, 1].set_xlabel("x in m")
ax[0, 1].grid()

ax[1, 0].plot(x, w_res[1:-1, :])
ax[1, 0].set_title("Water Content")
ax[1, 0].set_ylabel("w in kg/m³")
ax[1, 0].set_xlabel("x in m")
ax[1, 0].grid()

ax[1, 1].plot(x, pvap_res[1:-1, :])
ax[1, 1].set_title("Vapour Pressure")
ax[1, 1].set_ylabel("pvap in Pa")
ax[1, 1].set_xlabel("x in m")
ax[1, 1].grid()

fig.tight_layout()
fig.legend()
# fig.grid()

plt.show()
print("Done plotting.")
