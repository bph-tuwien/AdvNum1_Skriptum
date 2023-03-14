import matplotlib.pyplot as plt
import numpy as np


# %%
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

        self.RC = kwargs.get('RC', None)

        # Neighbours
        # No neighbours. This is the end of the sim domain


class Simulation(object):
    def __init__(self, whole_domain, mat_domain, sim_time):
        self.whole_domain = whole_domain
        self.mat_domain = mat_domain
        self.sim_time = sim_time

    def run_simulation(self):
        '''
        Function to run the simulation.
        :return: 
        '''

        sim_time = self.sim_time
        num_cells = whole_domain.__len__()

        # Apply neighbours
        for count, cell in enumerate(mat_domain):
            cell.cell_W = whole_domain[count]
            cell.cell_E = whole_domain[count + 2]

        dx_array = np.array([cell.dx for cell in mat_domain])

        #  Assemble Sim domain into matrices
        # Initialise variables

        Tn = np.array([cell.Tn for cell in whole_domain])
        Tn_plus = np.array([cell.Tn_plus for cell in whole_domain])
        # get interface indices
        interface_indices = [layer.__len__() for layer in layers]  # here the ghost cells are missing.
        RC_array = np.array([cell.RC for cell in whole_domain])
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
        K = K / cp_P_constants / rho_P_constants / dx_P_constants * dt
        K[0, 0], K[0, 1], K[-1, -1], K[-1, -2] = 0, 0, 0, 0  # ghost lines for boundary conditions
        # Rewrite [K] * [T] + [T] with identitiy matrix
        I = np.eye(num_cells, num_cells)
        L = I + K

        # Simulation Loop
        while t <= sim_time:
            # Calculate temperature field
            # ------------------------------
            # Tn_plus = K @ Tn + Tn
            Tn_plus = L @ Tn
            Tn = Tn_plus
            t += dt

        self.dx_array = dx_array
        self.Tn = Tn


        return

    def plot_results(self):
        '''
        Function to plot the results of the calculation.
        :return: 
        '''
        dx_array = self.dx_array
        d_array = np.cumsum(dx_array) - dx_array / 2

        plt.plot(d_array, Tn[1:-1], '.')
        plt.plot(d_array, Tn[1:-1], linestyle='dashed', alpha=0.2, color='b')
        # plt.xticks(np.arange(min(d_array), max(d_array), 0.01))
        for x in np.cumsum(dx_array[1:-1]):
            plt.axvline(x, alpha=0.2, color='black')
        plt.axvline(0, alpha=1, color='blue')
        plt.axvline(np.cumsum(dx_array)[-1], alpha=1, color='red')
        plt.annotate(xy=(10, 10), text='Left Boundary', fontsize=22)
        # plt.text(-5, 60, 'Right Boundary', fontsize=22)
        plt.text(0.02, 13, 'Left Boundary', style='italic', bbox={
            'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
        plt.text(0.26, 10, 'Right Boundary', style='italic', bbox={
            'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        plt.text(0.10, 2, 'Finite Volumes', style='italic', bbox={
            'facecolor': 'black', 'alpha': 0.3, 'pad': 10})

        plt.xlabel('Thickness in Meter')
        plt.ylabel('Temperature in Degree Celsius')
        plt.show()


# %%

eps = [0] * 20
eps = [Cell_Mat(dx=0.01, lamb=0.04, cp=1470, rho=1000, Tn=10) for i in eps]
concrete = [0] * 10
concrete = [Cell_Mat(dx=0.015, lamb=2.3, cp=1000, rho=2300, Tn=10) for i in concrete]
plaster = [0] * 4
plaster = [Cell_Mat(dx=0.005, lamb=2.3, cp=1000, rho=2000, Tn=10) for i in plaster]

# Define BCs
BC_left = [Boundary_Cell(Tn=0, RC=0.04)]
BC_right = [Boundary_Cell(Tn=20, RC=0.10)]

# Assemble simulation domain
layers = [eps, concrete, plaster]
mat_domain = eps + concrete + plaster
whole_domain = BC_left + mat_domain + BC_right


# %%

mySim1 = Simulation(whole_domain=whole_domain, mat_domain=mat_domain, sim_time=3600*24*1)
mySim1.run_simulation()
mySim1.plot_results()

