# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib as mpl

# mpl.use('Qt5Agg')  # interactive mode works with this, pick one
mpl.use('TkAgg')  # interactive mode works with this, pick one


# %% md
## Program Design

class Cell_Mat(object):
    def __init__(self, *args, **kwargs):
        # Discretization
        self.dx = kwargs.get('dx', None)
        self.face_e = kwargs.get('face_e', None)
        self.face_w = kwargs.get('face_w', None)

        # Material parameters
        self.lamb = kwargs.get('lamb', None)
        self.cp = kwargs.get('cp', None)
        self.rho = kwargs.get('rho', None)
        self.Lv = kwargs.get('Lv', None)
        # Because we are calculating the coupled problem, we need to use the lambda_w
        # instead of the lambda value

        self.mu = kwargs.get('mu', None)

        self.kw_lamb = kwargs.get('kw_lamb', None)
        self.kw_mu = kwargs.get('kw_mu', None)

        # Coupled parameters
        self._lambda_w = kwargs.get('lambda_w', None)
        self._mu_w = kwargs.get('delta_v_w', None)
        self._cm = kwargs.get('cm', None)
        self._cp_w = kwargs.get('cp_w', None)

        # Fields
        self.Tn = kwargs.get('Tn', None)
        self.Tn_plus = kwargs.get('Tn_plus', None)
        self.dT = kwargs.get('dT', None)
        self._pv = kwargs.get('pv', None)  # value for pv in Pascal
        self._w = kwargs.get('w', None)  # value for w in kg/m³
        self.phi = kwargs.get('phi', None)  # value for w in kg/m³
        self._psat = kwargs.get('psat', None)  # value for w in kg/m³

        # Neighbours
        self.cell_E = kwargs.get('cell_E', None)
        self.cell_W = kwargs.get('cell_W', None)

        # Calculated self values
        self._RC_T = kwargs.get('RC_T', None)
        self._RC_mu = kwargs.get('RC_mu', None)


    @property
    def pv(self):
        self.pv = self.phi * self.psat
        return self._pv

    @pv.setter
    def pv(self, value):
        self._pv = value

    @property
    def psat(self):
        self.psat = self.fc_pvsat(self.Tn)
        return self._psat

    @psat.setter
    def psat(self, value):
        self._psat = value

    @property
    def RC_T(self):
        self.RC_T = self.dx / self.lambda_w
        return self._RC_T

    @RC_T.setter
    def RC_T(self, value):
        self._RC_T = value

    @property
    def RC_mu(self):
        self.RC_mu = self.dx / self.mu_w
        return self._RC_mu

    @RC_mu.setter
    def RC_mu(self, value):
        self._RC_mu = value

    @property
    def w(self):
        self.w = self.fc_w_phi(self.phi)
        return self._w

    @w.setter
    def w(self, value):
        self._w = value

    # @property
    # def phi(self):
    #     # self.phi = self.pv / self.fc_pvsat(self.Tn) * 100
    #     self.phi = self.fc_w_phi(self.w)
    #     return self._phi
    #
    # @phi.setter
    # def phi(self, value):
    #     self._phi = value

    @property
    def cm(self):
        self.cm = self.update_Cm(self.w, self.phi, self.Tn)
        return self._cm

    @cm.setter
    def cm(self, value):
        self._cm = value

    @property
    def lambda_w(self):
        self.lambda_w = self.fc_lambda(self.w)
        return self._lambda_w

    @lambda_w.setter
    def lambda_w(self, value):
        self._lambda_w = value

    @property
    def mu_w(self):
        self.mu_w = self.fc_deltav(self.w)
        return self._mu_w

    @mu_w.setter
    def mu_w(self, value):
        self._mu_w = value

    @property
    def cp_w(self):
        self.cp_w = self.update_cp_w()
        return self._cp_w

    @cp_w.setter
    def cp_w(self, value):
        self._cp_w = value

    # Thermophysical functions
    # -----------------------------
    # vapour pressure at saturation
    def fc_pvsat(self, T):
        return np.power(10, 7.625 * T / (241 + T) + 2.7877)

    # vapour diffusivity depending on water content
    def fc_deltav(self, w):
        # a = 1.1 * 1e-7
        # b = -1.57 * 1e-7
        return self.mu + self.kw_mu * w

    # thermal conductivity depending on water content
    def fc_lambda(self, w):
        # a = 0.23
        # b = 6
        return self.lamb + self.kw_lamb * w

    # TODO: Adjust for each material
    # sorption curve
    def fc_w_phi(self, phi):
        a = 700  # 1000
        b = 250 * 1e-6  # 146*1e-6
        c = 2.9  # 1.59
        res = a / np.power(1 - np.log(phi / 100) / b, 1 / c)
        return res * 1e-3

    # sortpion curve the other way around
    def fc_phi_w(self, w):
        a = 700  # 1000
        b = 250 * 1e-6  # 146*1e-6
        c = 2.9  # 1.59
        # phi = np.zeros(len(w))
        phi = np.exp(b * (1 - np.power((a / (1000 * w)), c))) * 100
        return phi

    def update_cp_w(self):
        cp_w = self.cp + self.w / self.rho
        return cp_w

    def update_Cm(self, w, phi, T):
        # epsilon = 0.001 * np.min(w)
        # epsilon = 0.001 * w
        # wp = w + epsilon
        # phip = self.fc_phi_w(wp)
        # dw = abs(wp - w)
        # dphi = abs(phip - phi) / 100
        # pvs = self.fc_pvsat(T)
        # Cm = dw / dphi / pvs

        epsilon = 0.000001  # To avoid division by zero
        wE = self.cell_E.w
        wW = self.cell_W.w + epsilon
        phiE = self.cell_E.phi / 100
        phiW = self.cell_W.phi / 100 + epsilon
        Cm = np.abs((wE - wW) / (phiE - phiW)) * 1 / self.psat

        return Cm


class Boundary_Cell(object):
    def __init__(self, *args, **kwargs):
        # Fields
        self.Tn = kwargs.get('Tn', None)
        self.Tn_plus = kwargs.get('Tn_plus', None)
        self.pv = kwargs.get('pv', None)  # Vapour pressure in Pascal

        self.RC_T = kwargs.get('RC_T', None)
        self.RC_mu = kwargs.get('RC_mu', None)

        # Neighbours
        # No neighbours. This is the end of the sim domain


# ############################################################

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


# %% md
## Input data
# Here we create our input data for our calculation:
# %%
# ----------------------------------
# Define simulation domain
# ----------------------------------

# Define Simulation parameters
time_in_hours = 144 / 60  # in hours
time_in_hours = 1  # in hours
sim_time = 3600 * time_in_hours * 1
dt = 60  # in seconds

# Define mat cells
# eps = [0] * 20
# eps = [Cell_Mat(dx=0.01, lamb=0.04, mu=70, cp=1470, rho=1000, Tn=20, pv=1000, Lv=2400 * 1e3, kw_lamb=0.006,
#                 kw_mu=1.57 ** -10) for i in eps]
# concrete = [0] * 10
# concrete = [Cell_Mat(dx=0.015, lamb=2.3, mu=70, cp=1000, rho=2500, Tn=20, pv=1000, Lv=2400 * 1e3, kw_lamb=0.006,
#                      kw_mu=1.57 ** -10) for i in concrete]
# plaster = [0] * 4
# plaster = [Cell_Mat(dx=0.005, lamb=0.9, mu=70, cp=1000, rho=2000, Tn=20, pv=1000, Lv=2400 * 1e3, kw_lamb=0.006,
#                     kw_mu=1.57 ** -10) for i in plaster]

brick = [0] * 15
brick = [Cell_Mat(dx=0.1 / 15, lamb=0.23, mu=1.1 * 1e-7, cp=1000, rho=2800, Tn=20, pv=1000, phi=42.46, Lv=2400 * 1e3, kw_lamb=6,
                  kw_mu=-1.57 * 1e-7) for i in brick]

# Define BCs
# BC_left = [Boundary_Cell(Tn=20, RC_T=0.04, pv=1200, RC_mu=0.10)]
# BC_right = [Boundary_Cell(Tn=40, RC_T=0.10, pv=1500, RC_mu=0.10)]

BC_left = [Cell_Mat(dx=0.1 / 15, lamb=0.23, mu=1.1 * 1e-7, cp=1000, rho=2800, Tn=20, pv=1200, phi=51, Lv=2400 * 1e3, kw_lamb=6,
                    kw_mu=-1.57 * 1e-7)]
BC_right = [Cell_Mat(dx=0.1 / 15, lamb=0.23, mu=1.1 * 1e-7, cp=1000, rho=2800, Tn=20, pv=1200, phi=51, Lv=2400 * 1e3, kw_lamb=6,
                     kw_mu=-1.57 * 1e-7)]

# Assemble simulation domain
# layers = [eps, concrete, plaster]
layers = [brick]
# mat_domain = eps + concrete + plaster
mat_domain = brick
whole_domain = BC_left + mat_domain + BC_right
num_cells = whole_domain.__len__()

# %%
#  Assemble Sim domain into matrices
# Initialise variables

# Apply neighbours
for count, cell in enumerate(mat_domain):
    cell.cell_W = whole_domain[count]
    cell.cell_E = whole_domain[count + 2]

# %%
dx_array = np.array([cell.dx for cell in mat_domain])

# Define fields
Tn = np.array([cell.Tn for cell in whole_domain])
Tn_plus = np.array([cell.Tn_plus for cell in whole_domain])

w = np.array([cell.w for cell in whole_domain])


def update_conductivity_matrix_temp():
    RC_array = np.array([cell.RC_T for cell in whole_domain])
    RC_W = RC_array[0:-2]
    RC_P = RC_array[1:-1]
    RC_E = RC_array[2:]
    Kw_array = 1 / (RC_W / 2 + RC_P / 2)
    Ke_array = 1 / (RC_E / 2 + RC_P / 2)

    # Build conductivity matrix [K]
    # Create sparse matrix
    Left_diagonal = np.eye(num_cells, num_cells, k=-1)
    Left_diagonal[1:-1, 0:-2] = Left_diagonal[1:-1, 0:-2] * Kw_array
    Right_diagonal = np.eye(num_cells, num_cells, k=1)
    Right_diagonal[1:-1, 2:] = Right_diagonal[1:-1, 2:] * Ke_array
    Main_Diagonal = np.eye(num_cells, num_cells)
    Main_Diagonal[1:-1, 1:-1] = Main_Diagonal[1:-1, 1:-1] * - (Kw_array + Ke_array)
    K_T = Left_diagonal + Main_Diagonal + Right_diagonal

    # Multiply with constants
    # cpw_array = np.array([cell.cp_w for cell in mat_domain])
    cpw_array = np.array([cell.cp_w for cell in mat_domain])
    Left_diagonal = np.eye(num_cells, num_cells, k=-1)
    Left_diagonal[1:-1, 0:-2] = Left_diagonal[1:-1, 0:-2] * cpw_array
    Right_diagonal = np.eye(num_cells, num_cells, k=1)
    Right_diagonal[1:-1, 2:] = Right_diagonal[1:-1, 2:] * cpw_array
    Main_Diagonal = np.eye(num_cells, num_cells)
    Main_Diagonal[1:-1, 1:-1] = Main_Diagonal[1:-1, 1:-1] * cpw_array
    cp_P_constants = Left_diagonal + Main_Diagonal + Right_diagonal
    cp_P_constants[cp_P_constants == 0] = 1

    rho_array = np.array([cell.rho for cell in mat_domain])
    Left_diagonal = np.eye(num_cells, num_cells, k=-1)
    Left_diagonal[1:-1, 0:-2] = Left_diagonal[1:-1, 0:-2] * rho_array
    Right_diagonal = np.eye(num_cells, num_cells, k=1)
    Right_diagonal[1:-1, 2:] = Right_diagonal[1:-1, 2:] * rho_array
    Main_Diagonal = np.eye(num_cells, num_cells)
    Main_Diagonal[1:-1, 1:-1] = Main_Diagonal[1:-1, 1:-1] * rho_array
    rho_P_constants = Left_diagonal + Main_Diagonal + Right_diagonal
    rho_P_constants[rho_P_constants == 0] = 1

    Left_diagonal = np.eye(num_cells, num_cells, k=-1)
    Left_diagonal[1:-1, 0:-2] = Left_diagonal[1:-1, 0:-2] * dx_array
    Right_diagonal = np.eye(num_cells, num_cells, k=1)
    Right_diagonal[1:-1, 2:] = Right_diagonal[1:-1, 2:] * dx_array
    Main_Diagonal = np.eye(num_cells, num_cells)
    Main_Diagonal[1:-1, 1:-1] = Main_Diagonal[1:-1, 1:-1] * dx_array
    dx_P_constants = Left_diagonal + Main_Diagonal + Right_diagonal
    dx_P_constants[dx_P_constants == 0] = 1

    return K_T, cp_P_constants, rho_P_constants, dx_P_constants, cpw_array, rho_array


def update_conductivity_matrix_latent():
    RC_array = np.array([cell.RC_mu for cell in whole_domain])
    RC_W = RC_array[0:-2]
    RC_P = RC_array[1:-1]
    RC_E = RC_array[2:]
    Kw_array = 1 / (RC_W / 2 + RC_P / 2)
    Ke_array = 1 / (RC_E / 2 + RC_P / 2)

    # Build conductivity matrix [K]
    # Create sparse matrix
    Left_diagonal = np.eye(num_cells, num_cells, k=-1)
    Left_diagonal[1:-1, 0:-2] = Left_diagonal[1:-1, 0:-2] * Kw_array
    Right_diagonal = np.eye(num_cells, num_cells, k=1)
    Right_diagonal[1:-1, 2:] = Right_diagonal[1:-1, 2:] * Ke_array
    Main_Diagonal = np.eye(num_cells, num_cells)
    Main_Diagonal[1:-1, 1:-1] = Main_Diagonal[1:-1, 1:-1] * - (Kw_array + Ke_array)
    K_latent = Left_diagonal + Main_Diagonal + Right_diagonal

    latent_array = np.array([cell.Lv for cell in mat_domain])
    Left_diagonal = np.eye(num_cells, num_cells, k=-1)
    Left_diagonal[1:-1, 0:-2] = Left_diagonal[1:-1, 0:-2] * latent_array
    Right_diagonal = np.eye(num_cells, num_cells, k=1)
    Right_diagonal[1:-1, 2:] = Right_diagonal[1:-1, 2:] * latent_array
    Main_Diagonal = np.eye(num_cells, num_cells)
    Main_Diagonal[1:-1, 1:-1] = Main_Diagonal[1:-1, 1:-1] * latent_array
    latent_P_constants = Left_diagonal + Main_Diagonal + Right_diagonal
    latent_P_constants[latent_P_constants == 0] = 1

    return K_latent, latent_P_constants


def update_conductivity_matrix_pv():
    RC_array = np.array([cell.RC_mu for cell in whole_domain])
    RC_W = RC_array[0:-2]
    RC_P = RC_array[1:-1]
    RC_E = RC_array[2:]
    Kw_array = 1 / (RC_W / 2 + RC_P / 2)
    Ke_array = 1 / (RC_E / 2 + RC_P / 2)

    # Build conductivity matrix [K]
    # Create sparse matrix
    Left_diagonal = np.eye(num_cells, num_cells, k=-1)
    Left_diagonal[1:-1, 0:-2] = Left_diagonal[1:-1, 0:-2] * Kw_array
    Right_diagonal = np.eye(num_cells, num_cells, k=1)
    Right_diagonal[1:-1, 2:] = Right_diagonal[1:-1, 2:] * Ke_array
    Main_Diagonal = np.eye(num_cells, num_cells)
    Main_Diagonal[1:-1, 1:-1] = Main_Diagonal[1:-1, 1:-1] * - (Kw_array + Ke_array)
    K_mu = Left_diagonal + Main_Diagonal + Right_diagonal

    # Multiply with constants
    cm_array = np.array([cell.cm for cell in mat_domain])
    Left_diagonal = np.eye(num_cells, num_cells, k=-1)
    Left_diagonal[1:-1, 0:-2] = Left_diagonal[1:-1, 0:-2] * cm_array
    Right_diagonal = np.eye(num_cells, num_cells, k=1)
    Right_diagonal[1:-1, 2:] = Right_diagonal[1:-1, 2:] * cm_array
    Main_Diagonal = np.eye(num_cells, num_cells)
    Main_Diagonal[1:-1, 1:-1] = Main_Diagonal[1:-1, 1:-1] * cm_array
    cm_P_constants = Left_diagonal + Main_Diagonal + Right_diagonal
    cm_P_constants[cm_P_constants == 0] = 1

    return K_mu, cm_P_constants, cm_array


def fc_coupled_HAM(vec_Tp_wp, vec_T_w, dt):
    # split array
    n = int(len(vec_T_w) / 2)  # half index
    T, w = vec_T_w[0:n], vec_T_w[n:]
    Tp, wp = vec_Tp_wp[0:n], vec_Tp_wp[n:]

    # Update Instances
    for count, cell in enumerate(whole_domain):
        cell.Tn = Tp[count]
        cell.w = wp[count]

    phi = fc_phi_w(w)
    phi_p = fc_phi_w(wp)
    pv = phi / 100 * fc_pvsat(T)
    pvp = phi_p / 100 * fc_pvsat(Tp)

    K_T, cp_P_constants, rho_P_constants, dx_P_constants, cpw_array, rho_array = update_conductivity_matrix_temp()
    K_mu, cm_P_constants, cm_array = update_conductivity_matrix_pv()
    K_latent, latent_P_constants = update_conductivity_matrix_latent()

    K_T = build_explicit_matrix_temp(K_T, cp_P_constants, rho_P_constants, dx_P_constants, dt)
    K_mu = build_explicit_matrix_vap(K_mu, cm_P_constants, dx_P_constants, dt)
    K_Latent = build_explicit_matrix_latent(K_latent, cp_P_constants, latent_P_constants, dx_P_constants,
                                            rho_P_constants,
                                            dt)
    # Crank-Nicolson scheme
    # explicit and implicit parts for T
    exp_T = 0.5 * (np.dot(K_T, T) + np.dot(K_Latent, pv))
    imp_T = 0.5 * (np.dot(K_T, Tp) + np.dot(K_Latent, pvp))
    # without latent heat
    T_term = -Tp + T + exp_T + imp_T
    # explicit and implicit parts for pv
    exp_w = 0.5 * (np.dot(K_mu, pv))
    imp_w = 0.5 * (np.dot(K_mu, pvp))
    w_term = -wp + w + exp_w + imp_w
    # phi_term = fc_phi_w(w_term)
    # pv_term = phi_term / 100 * fc_pvsat(T_term)
    # send back as one array
    return np.hstack([T_term, w_term])


# %% md
### Simulation

# Before running the loop we will check for a suitable timestep with the **Fourier Number**.
# %%
# ----------------------
# Simulation
# ----------------------

K_T, cp_P_constants, rho_P_constants, dx_P_constants, cpw_array, rho_array = update_conductivity_matrix_temp()
K_mu, cm_P_constants, cm_array = update_conductivity_matrix_pv()

t = 0

# Fourier Number for temperature
smallest_dx = dx_array.min()
smallest_cp = cpw_array.min()
smallest_rho = rho_array.min()
lambda_w_array = np.array([cell.lambda_w for cell in mat_domain])
biggest_lamb = lambda_w_array.max()
dt_temp = 0.5 * smallest_cp * smallest_rho * smallest_dx ** 2 / biggest_lamb

# Fourier Number for vapor
smallest_dx = dx_array.min()
smallest_cm = cm_array.min()

mu_w_array = np.array([cell.mu_w for cell in mat_domain])
biggest_mu = mu_w_array.max()
dt_vap = 0.5 * smallest_cm * smallest_dx ** 2 / biggest_mu

# dt = min(dt_temp, dt_vap)

# Figures for real time plotting

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

dx_array = np.array([cell.dx for cell in mat_domain])
d_array = np.cumsum(dx_array) - dx_array / 2


# %% md
# Now we can multiply our conductivity matrix with the missing constants:
# %%
# Multiply [K] with constants
def build_explicit_matrix_temp(K, cp_w, rho, dx_P, dt):
    K = K / cp_w / rho / dx_P * dt
    K[0, 0], K[0, 1], K[-1, -1], K[-1, -2] = 0, 0, 0, 0  # ghost lines for boundary conditions
    # Rewrite [K] * [T] + [T] with identitiy matrix
    I = np.eye(num_cells, num_cells)
    L = I + K
    L = K
    return L


def build_explicit_matrix_vap(K, cm, dx_P, dt):
    # K = K / cm / dx_P * dt
    K = K / dx_P * dt
    K[0, 0], K[0, 1], K[-1, -1], K[-1, -2] = 0, 0, 0, 0  # ghost lines for boundary conditions
    # Rewrite [K] * [T] + [T] with identitiy matrix
    I = np.eye(num_cells, num_cells)
    M = I + K
    M = K
    return M


def build_explicit_matrix_latent(K, cp_w, latent_P_constants, dx_P, rho, dt):
    K = (K * latent_P_constants) / cp_w / rho / dx_P * dt
    K[0, 0], K[0, 1], K[-1, -1], K[-1, -2] = 0, 0, 0, 0  # ghost lines for boundary conditions
    # Rewrite [K] * [T] + [T] with identitiy matrix
    I = np.eye(num_cells, num_cells)
    Latent = I + K
    Latent = K
    return Latent


modulo_storage = int(0.1 * 3600)  # sim_time/dt/100
# preparing post-process
store_Text, store_pvext = [], []
store_w, store_phi, store_T, store_pv = [], [], [], []

# %% md
# And finally run our simulation.


# %%
# Simulation Loop
while t <= sim_time:
    # solve the coupled, non-linear system
    result_array = fsolve(fc_coupled_HAM,
                          np.hstack([Tn, w]),
                          args=(np.hstack([Tn, w]), dt))

    # split the result into temperature and vapor pressure
    Tn_plus = result_array[:num_cells]
    w_plus = result_array[num_cells:]

    Tn = Tn_plus
    w = w_plus
    phi = fc_phi_w(w)
    pv = phi / 100 * fc_pvsat(Tn)

    # do some storage for plotting
    if (int(t) % modulo_storage) == 0 and t != 0:
        store_T.append(Tn[1:-1])
        store_w.append(w[1:-1])
        store_phi.append(phi[1:-1])
        store_pv.append(pv[1:-1])


    t += dt

# %% md
## Plotting the results
# Let's plot our results.
# %%

print("\n#################\nPlotting (can be long)")
stop = int(len(store_pv))
start = 0

fig, axs = plt.subplots(2, 2)

#  Plot tempfield to check
plt.subplot(221)
dx_array = np.array([cell.dx for cell in mat_domain])
d_array = np.cumsum(dx_array) - dx_array / 2

# plt.plot(d_array, Tn[1:-1], '.')
# plt.plot(d_array, Tn[1:-1], linestyle='dashed', alpha=0.2, color='b')


for i in range(start, stop):
    plt.plot(d_array, store_T[i], linestyle='dashed', alpha=0.2, color='b')
    plt.plot(d_array, store_T[i], '.', label=f'{i}')

# plt.xticks(np.arange(min(d_array), max(d_array), 0.01))
for x in np.cumsum(dx_array):
    plt.axvline(x, alpha=0.2, color='black')
plt.axvline(0, alpha=1, color='blue')
plt.axvline(np.cumsum(dx_array)[-1], alpha=1, color='red')
plt.annotate(xy=(10, 10), text='Left Boundary', fontsize=22)
# plt.text(-5, 60, 'Right Boundary', fontsize=22)
plt.text(0.02, 15, f'Left Boundary={BC_left[0].Tn} °C', style='italic', bbox={
    'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
plt.text(0.21, 10, f'Right Boundary={BC_right[0].Tn} °C', style='italic', bbox={
    'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
plt.text(0.10, 2, 'Finite Volumes', style='italic', bbox={
    'facecolor': 'black', 'alpha': 0.3, 'pad': 10})
plt.title(f'Results for the temperature profile at t={sim_time / 3600} h')

plt.xlabel('Thickness in Meter')
plt.ylabel('Temperature in Degree Celsius')
plt.legend()
# plt.show()

#  Plot vapfield to check
plt.subplot(222)
dx_array = np.array([cell.dx for cell in mat_domain])
d_array = np.cumsum(dx_array) - dx_array / 2

# plt.plot(d_array, p_vap[1:-1], '.')
# plt.plot(d_array, p_vap[1:-1], linestyle='dashed', alpha=0.2, color='b')

for i in range(start, stop):
    plt.plot(d_array, store_pv[i], linestyle='dashed', alpha=0.2, color='b')
    plt.plot(d_array, store_pv[i], '.', label=f'{i}')

for x in np.cumsum(dx_array):
    plt.axvline(x, alpha=0.2, color='black')
plt.axvline(0, alpha=1, color='blue')
plt.axvline(np.cumsum(dx_array)[-1], alpha=1, color='red')
plt.annotate(xy=(10, 10), text='Left Boundary', fontsize=22)

plt.text(0.02, 15, f'Left Boundary={BC_left[0].Tn} °C', style='italic', bbox={
    'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
plt.text(0.21, 10, f'Right Boundary={BC_right[0].Tn} °C', style='italic', bbox={
    'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
plt.text(0.10, 2, 'Finite Volumes', style='italic', bbox={
    'facecolor': 'black', 'alpha': 0.3, 'pad': 10})
plt.title(f'Results for the Vapour-Pressure profile at t={sim_time / 3600} h')

plt.xlabel('Thickness in Meter')
plt.ylabel('Vapour Pressure in Pascal')
# plt.tight_layout()
plt.legend()

plt.show()

#  Plot rh to check
plt.subplot(223)
dx_array = np.array([cell.dx for cell in mat_domain])
d_array = np.cumsum(dx_array) - dx_array / 2

# plt.plot(d_array, p_vap[1:-1], '.')
# plt.plot(d_array, p_vap[1:-1], linestyle='dashed', alpha=0.2, color='b')

for i in range(start, stop):
    plt.plot(d_array, store_phi[i], linestyle='dashed', alpha=0.2, color='b')
    plt.plot(d_array, store_phi[i], '.', label=f'{i}')

for x in np.cumsum(dx_array):
    plt.axvline(x, alpha=0.2, color='black')
plt.axvline(0, alpha=1, color='blue')
plt.axvline(np.cumsum(dx_array)[-1], alpha=1, color='red')
plt.annotate(xy=(10, 10), text='Left Boundary', fontsize=22)

plt.text(0.02, 15, f'Left Boundary={BC_left[0].Tn} °C', style='italic', bbox={
    'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
plt.text(0.21, 10, f'Right Boundary={BC_right[0].Tn} °C', style='italic', bbox={
    'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
plt.text(0.10, 2, 'Finite Volumes', style='italic', bbox={
    'facecolor': 'black', 'alpha': 0.3, 'pad': 10})
plt.title(f'Results for the Relative Humidity at t={sim_time / 3600} h')

plt.xlabel('Thickness in Meter')
plt.ylabel('Relative Humidity in %')
# plt.tight_layout()
plt.legend()

plt.show()

#  Plot watfield to check
plt.subplot(224)
dx_array = np.array([cell.dx for cell in mat_domain])
d_array = np.cumsum(dx_array) - dx_array / 2

# plt.plot(d_array, p_vap[1:-1], '.')
# plt.plot(d_array, p_vap[1:-1], linestyle='dashed', alpha=0.2, color='b')

for i in range(start, stop):
    plt.plot(d_array, store_w[i], linestyle='dashed', alpha=0.2, color='b')
    plt.plot(d_array, store_w[i], '.', label=f'{i}')

for x in np.cumsum(dx_array):
    plt.axvline(x, alpha=0.2, color='black')
plt.axvline(0, alpha=1, color='blue')
plt.axvline(np.cumsum(dx_array)[-1], alpha=1, color='red')
plt.annotate(xy=(10, 10), text='Left Boundary', fontsize=22)

plt.text(0.02, 15, f'Left Boundary={BC_left[0].Tn} °C', style='italic', bbox={
    'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
plt.text(0.21, 10, f'Right Boundary={BC_right[0].Tn} °C', style='italic', bbox={
    'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
plt.text(0.10, 2, 'Finite Volumes', style='italic', bbox={
    'facecolor': 'black', 'alpha': 0.3, 'pad': 10})
plt.title(f'Results for the Water Content at t={sim_time / 3600} h')

plt.xlabel('Thickness in Meter')
plt.ylabel('Water Content in kg / m²')
plt.tight_layout()
plt.legend()

plt.show()

print('Done!')
