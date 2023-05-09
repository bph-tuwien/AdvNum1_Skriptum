import numpy as np

# TODO: Validate and correct still from chatgpt.

# Define the network topology
network = {
    'nodes': ['room1', 'room2', 'corridor1', 'corridor2'],
    'branches': [
        {'start': 'room1', 'end': 'corridor1', 'area': 1., 'length': 3.0},
        {'start': 'corridor1', 'end': 'room2', 'area': 1.0, 'length': 3},
        {'start': 'corridor1', 'end': 'corridor2', 'area': 1, 'length': 3},
        {'start': 'corridor2', 'end': 'room1', 'area': 1, 'length': 3},
        {'start': 'corridor2', 'end': 'room2', 'area': 1.0, 'length': 3},
    ]
}

# Define the boundary conditions
boundaries = {
    'room1': {'pressure': 100000.0, 'temperature': 20.0},
    'room2': {'pressure': 100000.0, 'temperature': 20.0},
    'corridor1': {'pressure': 100000.0, 'temperature': 20.0},
    'corridor2': {'pressure': 100000.0, 'temperature': 20.0}
}

# Define the airflow properties
airflow = {
    'density': 1.2,
    'viscosity': 1.8e-5
}

# Define the solver function
def solve_network(network, boundaries, airflow):
    # Initialize the matrix A and vector b
    n_nodes = len(network['nodes'])
    A = np.zeros((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    # Populate the matrix A and vector b
    for branch in network['branches']:
        start_idx = network['nodes'].index(branch['start'])
        end_idx = network['nodes'].index(branch['end'])
        alpha = airflow['density'] * branch['area'] / branch['length']
        beta = airflow['viscosity'] * branch['area'] / branch['length']
        A[start_idx][start_idx] += alpha + beta
        A[end_idx][end_idx] += alpha + beta
        A[start_idx][end_idx] -= alpha
        A[end_idx][start_idx] -= alpha
        b[start_idx] += alpha * boundaries[branch['start']]['pressure']
        b[end_idx] += alpha * boundaries[branch['end']]['pressure']

    # Solve the system of equations to obtain the pressures
    pressures = np.linalg.solve(A, b)

    # Calculate the mass flow rates
    mass_flows = {}
    for branch in network['branches']:
        start_idx = network['nodes'].index(branch['start'])
        end_idx = network['nodes'].index(branch['end'])
        alpha = airflow['density'] * branch['area'] / branch['length']
        beta = airflow['viscosity'] * branch['area'] / branch['length']
        mass_flow = alpha * (pressures[start_idx] - pressures[end_idx]) - beta * (1 / airflow['density'])
        mass_flows[branch['start'] + '-' + branch['end']] = mass_flow

    return pressures, mass_flows

# Solve the network and print the results
pressures, mass_flows = solve_network(network, boundaries, airflow)
print('Node pressures:')
for node, pressure in zip(network['nodes'], pressures):
    print(f'{node}: {pressure} Pa')
print('Branch mass flow rates:')
for branch, mass_flow in mass_flows.items():
    print(f'{branch}: {mass_flow} kg/s')
print('Done!')