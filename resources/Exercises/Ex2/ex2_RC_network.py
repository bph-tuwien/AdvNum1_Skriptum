import numpy as np
from numpy.linalg import inv, solve


# -----------------------------
#   Helper Classes
# -----------------------------

class Node(object):
    def __init__(self, Tn, index, *args, **kwargs):
        self.Tn = Tn    # in °C or K
        self._K = kwargs.get('K', None)
        self.index = index

    @property
    def K(self):
        if self._K is None:
            # self.K = [K for K in K_List if self.index == K.node_start or self.index == K.node_end]
            self.K = [K for K in K_List if self.index == K.node_start]
            self.K.append([K for K in K_BC_List if self.index == K.node_end][0])
        return self._K

    @K.setter
    def K(self, value):
        self._K = value


class K(object):
    def __init__(self, node_start, node_end, *args, **kwargs):
        # self.K_value = K_value
        self.cond = kwargs.get('cond', None)  # in W/K
        self.node_start = node_start
        self.node_end = node_end
        self._mass_flow = kwargs.get('mass_flow', None)  # in kg/s
        self._index = kwargs.get('index', None)
        self._K_value = kwargs.get('K_value', None)


    @property
    def index(self):
        if self._index is None:
            self.index = self.node_start + self.node_end
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def K_value(self):
        if self._K_value is None:
            self.K_value = self.cond + self.mass_flow * 1000    # specific heat capacity dry ar approx. 1000 J/kgK
        return self._K_value

    @K_value.setter
    def K_value(self, value):
        self._K_value = value

    @property
    def mass_flow(self):
        if self._mass_flow is None:
            self.mass_flow = 0
        return self._mass_flow

    @mass_flow.setter
    def mass_flow(self, value):
        self._mass_flow = value


class Source(object):
    def __init__(self, index, heatflow, *args, **kwargs):
        self.index = index
        self.heatflow = heatflow


# %%

if __name__ == '__main__':

    # Create nodes and conductances
    # Main nodes
    T1 = Node(Tn=20, index='1')
    T2 = Node(Tn=20, index='2')
    T3 = Node(Tn=20, index='3')
    # Node_List = [T1, T2]
    Node_List = [T1, T2, T3]

    # Boundaries
    T01 = Node(Tn=20, index='01')
    T02 = Node(Tn=20, index='02')
    T03 = Node(Tn=20, index='03')
    # BC_Node_List = [T01, T02]
    BC_Node_List = [T01, T02, T03]

    # Conductances
    # K12 = K(cond=0.1, mass_flow=0.05, node_start='1', node_end='2')
    K12 = K(cond=0.1, mass_flow=0, node_start='1', node_end='2')
    K21 = K(cond=0.1, mass_flow=0, node_start='2', node_end='1')
    K13 = K(cond=0.1, mass_flow=0, node_start='1', node_end='3')
    K31 = K(cond=0.1, mass_flow=0, node_start='3', node_end='1')
    K23 = K(cond=0.1, mass_flow=0, node_start='2', node_end='3')
    # K32 = K(cond=0.1, mass_flow=0.05, node_start='3', node_end='2')
    K32 = K(cond=0.1, mass_flow=0, node_start='3', node_end='2')

    # BC Conductances
    K01 = K(K_value=0.1, node_start='01', node_end='1')
    K02 = K(K_value=0.1, node_start='02', node_end='2')
    K03 = K(K_value=0.1, node_start='03', node_end='3')
    K_List = [K12, K21, K13, K31, K23, K32, K01, K02, K03]
    # K_List = [K12, K21, K01, K02]
    K_BC_List = [K01, K02, K03]
    # K_BC_List = [K01, K02]

    # Sources
    I1 = Source(index='1', heatflow=10)
    # I1 = Source(index='1', heatflow=0)
    I2 = Source(index='2', heatflow=20)
    # I2 = Source(index='2', heatflow=0)
    I3 = Source(index='3', heatflow=10)
    # I3 = Source(index='3', heatflow=0)

    Sources_List = [I1, I2, I3]
    # Sources_List = [I1, I2]

    # End User Input
    # -------------------------------------------------------
    # %%
    K_index_list = [K.index for K in K_List]
    K_value_list = [K.K_value for K in K_List]
    K_value_dict = dict(
    zip(K_index_list, K_value_list))  # Dictionary where the K values are stored with their index as keys.

    # -----------------------------
    #         NUMERICS
    # -----------------------------

    # Stationary Calculation
    unknowns = Node_List.__len__()
    # Temperature Vector
    Tn = np.array([Node.Tn for Node in Node_List])

    # Conductivity Matrix
    main_diag_k = np.zeros(Tn.size)  # Be careful when using zero- or one-like they are bound to the dtype of the copied
    # array. Eg. if you are only putitng int as an initial condition into the array you copy numpy will assign the
    # dtype int.ö The copy will only accept int or round to int.

    # TODO: Check if this can be done smarter
    for index, node in enumerate(Node_List):
        main_diag_k[index] = np.sum([K.K_value for K in node.K])
    Kt_Array = np.eye(unknowns, unknowns) * -main_diag_k

    for index, elem in np.ndenumerate(Kt_Array):
        i = index[0]  # Row index
        j = index[1]  # Column index
        str_index = str(i + 1) + str(j + 1)
        if str_index in K_value_dict.keys():
            # Kt_Array[i, j] = K_value_dict[str_index]
            Kt_Array[j, i] = K_value_dict[str_index]  # Symmetrical Matrix

    # Boundary Conductivity Matrix
    num_bc_nodes = BC_Node_List.__len__()
    num_k_bc = K_BC_List.__len__()
    K_BC_Array = np.zeros((unknowns, num_bc_nodes), dtype=object)
    K_BC_Array[:] = [K for K in K_BC_List]
    K_value_BC_Array = np.zeros((unknowns, num_bc_nodes))
    K_value_BC_Array[:] = [K.K_value for K in K_BC_List]

    K_BC_Adjacency = np.zeros((unknowns, num_bc_nodes))
    for i, node in enumerate(Node_List):
        for j, K in np.ndenumerate(K_BC_Array):
            if node.index == K.node_start or node.index == K.node_end:
                K_BC_Adjacency[i, j[1]] = 1
    # Sources
    I_sources = np.zeros(unknowns)
    for source in Sources_List:
        idx = int(source.index) - 1  # adjust index by subtracting 1
        I_sources[idx] = source.heatflow

    # Boundary Temperatures
    T_BC_vec = np.array([node.Tn for node in BC_Node_List])
    K_BC = K_BC_Adjacency * K_value_BC_Array
    Io = K_BC @ T_BC_vec + I_sources

    # Calculation
    print('Solving Stationary Calculation')
    # T = -inv(Kt_Array) @ Io
    T = solve(Kt_Array, -Io)

    print(f'The temperature vector is: {T} ')
    print('End of Stationary Calculation')