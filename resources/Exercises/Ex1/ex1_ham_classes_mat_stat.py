import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, minimize


# %%
# TODO: Change this to layer and give discretization as input
class Cell_Mat(object):
    def __init__(self, *args, **kwargs):
        # Discretization
        # self.t = kwargs.get('t', None)
        # self.num_x = kwargs.get('num_x', None)
        self.dx = kwargs.get('dx', None)
        self.face_e = kwargs.get('face_e', None)
        self.face_w = kwargs.get('face_w', None)

        # Material parameters
        self.lamb = kwargs.get('lamb', None)
        self.cp = kwargs.get('cp', None)
        self.rho = kwargs.get('rho', None)
        self.RC = self.dx / self.lamb

        # Fields
        self.Tn = kwargs.get('Tn', None)
        self.Tn_plus = kwargs.get('Tn_plus', None)
        self.dT = kwargs.get('dT', None)

        # Neighbours
        self.cell_E = kwargs.get('cell_E', None)
        self.cell_W = kwargs.get('cell_W', None)


class Boundary_Cell(object):
    def __init__(self, *args, **kwargs):
        # Fields
        self.Tn = kwargs.get('Tn', None)
        self.Tn_plus = kwargs.get('Tn_plus', None)

        self.RC = kwargs.get('RC_T', None)

        # Neighbours
        # No neighbours. This is the end of the sim domain


class Simulation(object):
    def __init__(self, whole_domain, *args, **kwargs):
        self.whole_domain = kwargs.get('whole_domain', None)


# %%
if __name__ == '__main__':

    # ----------------------------------
    # Define simulation domain
    # ----------------------------------

    # Define Simulation parameters
    end_time = 3600 * 24 * 1
    dt = 1

    # Define mat cells
    eps = [0] * 20
    eps = [Cell_Mat(dx=0.01, lamb=0.04, cp=1470, rho=1040, Tn=10) for i in eps]
    concrete = [0] * 10
    concrete = [Cell_Mat(dx=0.015, lamb=2.3, cp=1000, rho=2500, Tn=10) for i in concrete]
    plaster = [0] * 4
    plaster = [Cell_Mat(dx=0.005, lamb=2.3, cp=1000, rho=2500, Tn=10) for i in plaster]

    # Define BCs
    BC_left = [Boundary_Cell(Tn=0, RC=0.04)]
    BC_right = [Boundary_Cell(Tn=20, RC=0.10)]

    # Assemble simulation domain
    layers = [eps, concrete, plaster]
    mat_domain = eps + concrete + plaster
    whole_domain = BC_left + mat_domain + BC_right

    num_cells = whole_domain.__len__()

    # Apply neighbours
    for count, cell in enumerate(mat_domain):
        cell.cell_W = whole_domain[count]
        cell.cell_E = whole_domain[count + 2]

    # %%
    #  Plot sim-domain to check
    dx_array = np.array([cell.dx for cell in mat_domain])
    d_array = np.cumsum(dx_array)
    dot_array = np.ones_like(d_array)

    plt.plot(d_array, dot_array, '.')
    plt.show()

    # %%
    #  Assemble Sim domain into matrices
    # Initialise variables

    Tn = np.array([cell.Tn for cell in whole_domain])
    Tn_plus = np.array([cell.Tn_plus for cell in whole_domain])
    # get interface indices
    interface_indices = [layer.__len__() for layer in layers]  # here the ghost cells are missing.
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
    K = Left_diagonal + Main_Diagonal + Right_diagonal
    # Multiply with constants
    cp_array = np.array([cell.cp for cell in mat_domain])
    Left_diagonal = np.eye(num_cells, num_cells, k=-1)
    Left_diagonal[1:-1, 0:-2] = Left_diagonal[1:-1, 0:-2] * cp_array
    Right_diagonal = np.eye(num_cells, num_cells, k=1)
    Right_diagonal[1:-1, 2:] = Right_diagonal[1:-1, 2:] * cp_array
    Main_Diagonal = np.eye(num_cells, num_cells)
    Main_Diagonal[1:-1, 1:-1] = Main_Diagonal[1:-1, 1:-1] * cp_array
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


    # %%

    # ----------------------
    # Simulation
    # ----------------------
    t = 0
    smallest_dx = dx_array.min()
    smallest_cp = cp_array.min()
    smallest_rho = rho_array.min()
    lamb_array = np.array([cell.lamb for cell in mat_domain])
    biggest_lamb = lamb_array.max()
    dt = 0.5 * smallest_cp * smallest_rho * smallest_dx ** 2 / biggest_lamb

    # Multiply [K] with constants
    K = K / dx_P_constants
    K[0, 0], K[0, 1], K[-1, -1], K[-1, -2] = 0, 0, 0, 0  # ghost lines for boundary conditions

    T_stat = np.copy(Tn)
    def constraint_BC_left(T_stat):
        bc_left = T_stat[0]
        return bc_left - 0

    def constraint_BC_right(T_stat):
        bc_right = T_stat[-1]
        return bc_right - 20


    cons = [{'type': 'eq', 'fun': constraint_BC_left},
            {'type': 'eq', 'fun': constraint_BC_right}]


    def temp_stat(K, T_stat):
        return K @ T_stat

    T_stat = minimize(temp_stat, Tn, args=(K), constraints=cons)



    # %%
    #  Plot tempfield to check
    dx_array = np.array([cell.dx for cell in mat_domain])
    d_array = np.cumsum(dx_array)
    temp_array = np.array([cell.Tn for cell in mat_domain])

    plt.plot(d_array, Tn[1:-1])
    plt.show()
