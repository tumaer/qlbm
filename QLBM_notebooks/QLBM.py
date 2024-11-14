import numpy as np
import qiskit
from qiskit.circuit.library import HGate, XGate
from qiskit import QuantumRegister, QuantumCircuit

def InitializeQC(density_field: np.ndarray, num_velocities: int) -> QuantumCircuit:
    """
    Initializes a quantum circuit based on the input density field and the number of velocities.

    Parameters:
    - density_field (np.ndarray): The density field as a numpy array, which can be 1D, 2D, or 3D.
    - num_velocities (int): The number of possible velocities in the simulation, used to determine the number of qubits for velocity encoding.

    Returns:
    - QuantumCircuit: A quantum circuit with initialized qubits based on the flattened and normalized density field.
    """
    
    # Determine dimensions and shape of the input field
    dimensions = density_field.ndim
    shape = density_field.shape
    
    # Calculate the required number of qubits for encoding velocities
    num_qubits_velocities = int(np.ceil(np.log2(num_velocities)))
    num_ancilla_qubits = 1  # Single ancilla qubit
    
    # Flatten and normalize the density field
    density_field_flattened = density_field.flatten(order='F')
    density_field_normalized = density_field_flattened / np.linalg.norm(density_field_flattened)
    
    # Define spatial quantum registers based on the number of dimensions (x, y, z)
    spatial_registers = []
    if dimensions >= 1:
        spatial_registers.append(QuantumRegister(int(np.log2(shape[0])), 'qx'))
    if dimensions >= 2:
        spatial_registers.append(QuantumRegister(int(np.log2(shape[1])), 'qy'))
    if dimensions == 3:
        spatial_registers.append(QuantumRegister(int(np.log2(shape[2])), 'qz'))
    
    # Create registers for velocities and ancilla
    velocity_register = QuantumRegister(num_qubits_velocities, 'qf')
    ancilla_register = QuantumRegister(num_ancilla_qubits, 'anc')
    
    # Combine all registers into the quantum circuit
    qc = QuantumCircuit(*spatial_registers, velocity_register, ancilla_register)
    
    # Initialize the circuit with the normalized density field, targeting the spatial qubits
    qc.initialize(density_field_normalized, [q for reg in spatial_registers for q in reg])
    
    # Add a barrier to separate initialization from subsequent operations
    qc.barrier()
    
    return qc

def duplicate_density_field(qc: QuantumCircuit, num_velocities: int) -> None:
    """
    Duplicates the density field in a quantum circuit based on the specified number of velocities.
    
    Parameters:
    - qc (QuantumCircuit): Quantum circuit on which the density field duplication will be applied.
    - num_velocities (int): Specifies the number of velocities in the simulation, which determines the duplication pattern.
    
    Applies a sequence of Hadamard and controlled gates to duplicate the density field according to specific D1Q3, D2Q5, D2Q9, and D3Q27 models.
    """
    
    # D1Q3 model duplication
    if num_velocities == 3:
        qc.h(qc.qregs[1][0])                         # Apply Hadamard on qf[0]
        qc.ch(qc.qregs[1][0], qc.qregs[1][1])        # Controlled-Hadamard from qf[0] to qf[1]
        qc.cx(qc.qregs[1][1], qc.qregs[1][0])        # CX between qf[1] and qf[0]
    
    # D2Q5 model duplication
    elif num_velocities == 5:
        mc2_h = HGate().control(2)                   # Two-controlled Hadamard gate
        qc.h(qc.qregs[2][0])
        qc.h(qc.qregs[2][1])
        qc.x(qc.qregs[2][0])
        qc.append(mc2_h, [qc.qregs[2][0], qc.qregs[2][1], qc.qregs[2][2]])
        qc.mcx([qc.qregs[2][0], qc.qregs[2][2]], qc.qregs[2][1])
        qc.x(qc.qregs[2][0])
    
    # D2Q9 model duplication
    elif num_velocities == 9:
        mc1_h = HGate().control(1)                   # One-controlled Hadamard gate
        mc3_h = HGate().control(3)                   # Three-controlled Hadamard gate
        mc3_x = XGate().control(3)                   # Three-controlled X gate
        qc.h(qc.qregs[2][0])
        qc.h(qc.qregs[2][1])
        qc.append(mc1_h, [qc.qregs[2][1], qc.qregs[2][2]])
        qc.cx(qc.qregs[2][2], qc.qregs[2][1])
        qc.append(mc1_h, [qc.qregs[2][2], qc.qregs[2][1]])
        qc.x(qc.qregs[2][2])
        qc.x(qc.qregs[2][1])
        qc.append(mc3_h, [qc.qregs[2][0], qc.qregs[2][1], qc.qregs[2][2], qc.qregs[2][3]])
        qc.append(mc3_x, [qc.qregs[2][3], qc.qregs[2][1], qc.qregs[2][2], qc.qregs[2][0]])
        qc.x(qc.qregs[2][2])
        qc.x(qc.qregs[2][1])
    
    # D3Q27 model duplication
    elif num_velocities == 27:
        mc3_h = HGate().control(3)                   # Three-controlled Hadamard gate
        mc4_h = HGate().control(4)                   # Four-controlled Hadamard gate
        qc.h(qc.qregs[3][0])
        qc.h(qc.qregs[3][1])
        qc.h(qc.qregs[3][2])
        qc.ch(qc.qregs[3][2], qc.qregs[3][3])
        qc.cx(qc.qregs[3][3], qc.qregs[3][2])
        qc.ch(qc.qregs[3][2], qc.qregs[3][3])
        qc.ch(qc.qregs[3][3], qc.qregs[3][4])
        qc.cx(qc.qregs[3][4], qc.qregs[3][3])
        qc.mcx([qc.qregs[3][4], qc.qregs[3][2], qc.qregs[3][1]], qc.qregs[3][3])
        qc.x(qc.qregs[3][1])
        qc.append(mc3_h, [qc.qregs[3][2], qc.qregs[3][3], qc.qregs[3][4], qc.qregs[3][1]])
        qc.x(qc.qregs[3][1])
        qc.mcx([qc.qregs[3][4], qc.qregs[3][2], qc.qregs[3][1]], qc.qregs[3][3])
        qc.mcx([qc.qregs[3][4], qc.qregs[3][3]], qc.qregs[3][2])
        qc.x(qc.qregs[3][1])
        qc.mcx([qc.qregs[3][4], qc.qregs[3][3], qc.qregs[3][1], qc.qregs[3][0]], qc.qregs[3][2])
        qc.x(qc.qregs[3][1])
        qc.mcx([qc.qregs[3][4], qc.qregs[3][3], qc.qregs[3][2], qc.qregs[3][0]], qc.qregs[3][1])
        qc.x(qc.qregs[3][0])
        qc.append(mc4_h, [qc.qregs[3][4], qc.qregs[3][3], qc.qregs[3][1], qc.qregs[3][2], qc.qregs[3][0]])
        qc.x(qc.qregs[3][0])
        qc.mcx([qc.qregs[3][4], qc.qregs[3][3], qc.qregs[3][1]], qc.qregs[3][2])
        qc.x(qc.qregs[3][2])
        qc.mcx([qc.qregs[3][4], qc.qregs[3][3], qc.qregs[3][2]], qc.qregs[3][1])
        qc.x(qc.qregs[3][2])

    # Add a barrier after duplication for circuit readability
    qc.barrier()


def collision(
        qc: QuantumCircuit, 
        velocity_field: np.array, 
        num_velocities: int
        ) -> None:
    """
    Applies the collision operator to a quantum circuit based on the specified velocity field 
    and number of velocities in a lattice model.

    Parameters:
    - qc (QuantumCircuit): Quantum circuit where the collision will be applied.
    - velocity_field (np.array): Numpy array representing the velocity field of the fluid.
     SHAPE 1D: [N_x,1], SHAPE 2D: [N_x,N_y,2], SHAPE 3D: [N_x,N_y,N_z,3]
    - num_velocities (int): Number of velocities in the lattice model (e.g., 3, 5, 9, 27).
    """
    
    c_s = 1 / np.sqrt(3)  # Speed of sound constant in lattice models

    # D1Q3 collision model
    if num_velocities == 3:
        velocity_field_flattened = velocity_field.flatten(order='F')
        renorm_const = [np.sqrt(2), 2, 2]
        weights = [4/6, 1/6, 1/6]
        
        # Calculate distribution functions f_0, f_1, and f_2
        f_0 = weights[0] * renorm_const[0] * np.ones(len(velocity_field_flattened))
        f_1 = weights[1] * renorm_const[1] * (1 + velocity_field_flattened / c_s**2)
        f_2 = weights[2] * renorm_const[2] * (1 - velocity_field_flattened / c_s**2)
        
        f = np.concatenate((f_0, f_1, f_2))
        U_1 = f + 1j * np.sqrt(1 - np.square(f))
        U_2 = f - 1j * np.sqrt(1 - np.square(f))
        Collision_diagonal_entries = np.concatenate((U_1, np.ones(velocity_field.size), U_2, np.ones(velocity_field.size))).tolist()
        
        # Apply the collision operation
        qc.h(qc.qregs[-1])  # Apply Hadamard to ancilla qubit
        qc.diagonal(Collision_diagonal_entries, qc.qubits)
        qc.h(qc.qregs[-1])

    # D2Q5 collision model
    elif num_velocities == 5:
        velocity_field_flattened = velocity_field.flatten(order='F')
        renorm_const = [2, 2, 2 * np.sqrt(2), 2, 2 * np.sqrt(2)]
        weights = [1/3, 1/6, 1/6, 1/6, 1/6]
        
        lattice_velocities = get_velocity_set(num_velocities)
        N_x, N_y, _ = velocity_field.shape
        f = np.zeros([N_x, N_y, num_velocities])
        
        for jj in range(num_velocities):
            f[:, :, jj] = weights[jj] * renorm_const[jj] * (1 + (velocity_field[:, :, 0] * lattice_velocities[0, jj] + velocity_field[:, :, 1] * lattice_velocities[1, jj]) / c_s**2)
        
        f = f.flatten(order='F')
        U_1 = f + 1j * np.sqrt(1 - np.square(f))
        U_2 = f - 1j * np.sqrt(1 - np.square(f))
        Collision_diagonal_entries = np.concatenate([U_1, np.ones(2 ** (len(qc.qregs[0]) + len(qc.qregs[1])) * 3), U_2, np.ones(2 ** (len(qc.qregs[0]) + len(qc.qregs[1])) * 3)]).tolist()
        
        # Apply the collision operation
        qc.h(qc.qregs[-1])
        qc.diagonal(Collision_diagonal_entries, qc.qubits)
        qc.h(qc.qregs[-1])

    # D2Q9 collision model
    elif num_velocities == 9:
        velocity_field_flattened = velocity_field.flatten(order='F')
        renorm_const = [2, 2 * np.sqrt(2), 2 * np.sqrt(2), 2 * np.sqrt(2), 4, 4, 4, 4, 2 * np.sqrt(2)]
        weights = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
        
        lattice_velocities = get_velocity_set(num_velocities)
        N_x, N_y, _ = velocity_field.shape
        f = np.zeros([N_x, N_y, num_velocities])
        
        for jj in range(num_velocities):
            f[:, :, jj] = weights[jj] * renorm_const[jj] * (1 + (velocity_field[:, :, 0] * lattice_velocities[0, jj] + velocity_field[:, :, 1] * lattice_velocities[1, jj]) / c_s**2)
        
        f = f.flatten(order='F')
        U_1 = f + 1j * np.sqrt(1 - np.square(f))
        U_2 = f - 1j * np.sqrt(1 - np.square(f))
        Collision_diagonal_entries = np.concatenate((U_1, np.ones(2 ** (len(qc.qregs[0]) + len(qc.qregs[1])) * 7), U_2, np.ones(2 ** (len(qc.qregs[0]) + len(qc.qregs[1])) * 7))).tolist()
        
        # Apply the collision operation
        qc.h(qc.qregs[-1])
        qc.diagonal(Collision_diagonal_entries, qc.qubits)
        qc.h(qc.qregs[-1])

    # D3Q27 collision model
    elif num_velocities == 27:
        velocity_field_flattened = velocity_field.flatten(order='F')
        renorm_const = np.ones(27)
        renorm_const[0:4] = np.sqrt(2)**3
        renorm_const[4:12] = np.sqrt(2)**5
        renorm_const[12:16] = np.sqrt(2)**6
        renorm_const[16:20] = np.sqrt(2)**5
        renorm_const[20:22] = np.sqrt(2)**6
        renorm_const[22:24] = np.sqrt(2)**7
        renorm_const[24:26] = np.sqrt(2)**8
        renorm_const[26:27] = np.sqrt(2)**7
        weights = [8/27, 2/27, 2/27, 2/27, 2/27, 2/27, 2/27, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216]
        
        lattice_velocities = get_velocity_set(num_velocities)
        N_x, N_y, N_z, _ = velocity_field.shape
        f = np.zeros([N_x, N_y, N_z, num_velocities])
        
        for jj in range(num_velocities):
            f[:, :, :, jj] = weights[jj] * renorm_const[jj] * (1 + (velocity_field[:, :, :, 0] * lattice_velocities[0, jj] + velocity_field[:, :, :, 1] * lattice_velocities[1, jj] + velocity_field[:, :, :, 2] * lattice_velocities[2, jj]) / c_s**2)
        
        f = f.flatten(order='F')
        U_1 = f + 1j * np.sqrt(1 - np.square(f))
        U_2 = f - 1j * np.sqrt(1 - np.square(f))
        Collision_diagonal_entries = np.concatenate((U_1, np.ones(2 ** (len(qc.qregs[0]) + len(qc.qregs[1]) + len(qc.qregs[2])) * 5), U_2, np.ones(2 ** (len(qc.qregs[0]) + len(qc.qregs[1]) + len(qc.qregs[2])) * 5))).tolist()
        
        # Apply the collision operation
        qc.h(qc.qregs[-1])
        qc.diagonal(Collision_diagonal_entries, qc.qubits)
        qc.h(qc.qregs[-1])
    
    # Add a barrier for separation
    qc.barrier()


def get_binary_encoding_of_distribution_functions_gate(
        qc: QuantumCircuit,num_microscopic_qubits: int,
        lattice_index: int,
        index_first_microscopic_qubit: int
        ) -> None:
    """
    Encodes the binary representation of a given lattice index onto specified qubits in the quantum circuit with X gates.
    
    Parameters:
    - qc (QuantumCircuit): The quantum circuit where the encoding will be applied.
    - num_microscopic_qubits (int): The number of qubits representing the microscopic distribution functions.
    - lattice_index (int): The index to be encoded in binary format onto the qubits.
    - index_first_microscopic_qubit (int): The starting index of the qubits in the circuit where the encoding will be applied.
    
    Converts the lattice index to a binary representation and applies X gates to set the target qubits to `0`
    where the binary representation has `0`s, effectively encoding the binary pattern of `lattice_index`.
    """
    
    # Convert lattice index to a binary list with the specified number of qubits, in reverse order
    lattice_index_as_binary_list = [int(d) for d in str(format(lattice_index, f"0{num_microscopic_qubits}b"))]
    lattice_index_as_binary_list.reverse()
    # Apply X gate to the qubits corresponding to 0s in the binary representation
    for i, binary_value in enumerate(lattice_index_as_binary_list):
        if binary_value == 0:
            qc.x(index_first_microscopic_qubit + i)

def get_right_shift_gate(
    qc: QuantumCircuit,
    index_first_qubit_of_dimension: list,
    num_qubits_per_dimension: list,
    lattice_index: int,
    dimension_index: int,
    num_microscopic_qubits: int,
    index_first_microscopic_qubit: int,
    index_last_microscopic_qubits: int
) -> None:
    """
    Constructs a right shift gate in the quantum circuit, using multi-controlled Toffoli gates, to shift qubits in a specified dimension.

    Parameters:
    - qc (QuantumCircuit): The quantum circuit to which the right shift gate will be applied.
    - index_first_qubit_of_dimension (list): List of starting qubit indices for each dimension.
    - num_qubits_per_dimension (list): List specifying the number of qubits in each dimension.
    - lattice_index (int): Lattice index encoded as a binary value in the microscopic qubits.
    - dimension_index (int): The index of the dimension along which the shift will occur.
    - num_microscopic_qubits (int): The total number of microscopic qubits.
    - index_first_microscopic_qubit (int): Starting index of the microscopic qubits.
    - index_last_microscopic_qubits (int): Last index of the microscopic qubits.


    This function performs a controlled right shift on qubits within a specified dimension by applying multi-controlled Toffoli gates.
    The shift is based on the binary encoding of `lattice_index`.
    """
    
    # Identify the qubits involved in the specified dimension
    qubits_to_shift = range(
        index_first_qubit_of_dimension[dimension_index],
        index_first_qubit_of_dimension[dimension_index] + num_qubits_per_dimension[dimension_index]
    )
    
    # Convert lattice index to binary and reverse it for correct qubit mapping
    binary_representation = [int(bit) for bit in f"{lattice_index:0{num_microscopic_qubits}b}"]
    binary_representation.reverse()
    
    # Identify control qubits for the microscopic encoding
    microscopic_control_qubits = range(index_first_microscopic_qubit, index_last_microscopic_qubits)
    
    # Apply multi-controlled Toffoli gates for the right shift operation
    for target_qubit in reversed(qubits_to_shift):
        control_qubits = list(range(qubits_to_shift[0], target_qubit))
        qc.mct(control_qubits + list(microscopic_control_qubits), target_qubit)


def get_left_shift_gate(
    qc: QuantumCircuit,
    index_first_qubit_of_dimension: list,
    num_qubits_per_dimension: list,
    lattice_index: int,
    dimension_index: int,
    num_microscopic_qubits: int,
    index_first_microscopic_qubit: int,
    index_last_microscopic_qubits: int
) -> None:
    """
    Constructs a left shift gate in the quantum circuit, using multi-controlled Toffoli gates, to shift qubits in a specified dimension.

    Parameters:
    - qc (QuantumCircuit): The quantum circuit where the left shift gate will be applied.
    - index_first_qubit_of_dimension (list): List of starting qubit indices for each dimension.
    - num_qubits_per_dimension (list): List specifying the number of qubits in each dimension.
    - lattice_index (int): Lattice index encoded as a binary value in the microscopic qubits.
    - dimension_index (int): The index of the dimension along which the shift will occur.
    - num_microscopic_qubits (int): The total number of microscopic qubits.
    - index_first_microscopic_qubit (int): Starting index of the microscopic qubits.
    - index_last_microscopic_qubits (int): Last index of the microscopic qubits.

    Description:
    - This function performs a controlled left shift on qubits within a specified dimension by applying multi-controlled Toffoli gates.
      The shift is based on the binary encoding of `lattice_index`.
    """
    
    # Identify the qubits involved in the specified dimension
    qubits_to_shift = range(
        index_first_qubit_of_dimension[dimension_index],
        index_first_qubit_of_dimension[dimension_index] + num_qubits_per_dimension[dimension_index]
    )
    
    # Convert lattice index to binary and reverse it for correct qubit mapping
    binary_representation = [int(bit) for bit in f"{lattice_index:0{num_microscopic_qubits}b}"]
    binary_representation.reverse()
    
    # Identify control qubits for the microscopic encoding
    microscopic_control_qubits = range(index_first_microscopic_qubit, index_last_microscopic_qubits)
    
    # Apply multi-controlled Toffoli gates for the left shift operation
    for target_qubit in qubits_to_shift:
        control_qubits = list(range(qubits_to_shift[0], target_qubit))
        qc.mct(list(microscopic_control_qubits) + control_qubits, target_qubit)
        

def streaming(
        qc: QuantumCircuit, 
        num_velocities: int
        ) -> None:
    """
    Performs the streaming operation in a quantum circuit for a lattice-based model based on the number of velocities.
    
    Parameters:
    - qc (QuantumCircuit): The quantum circuit where the streaming operation will be applied.
    - num_velocities (int): The number of velocities in the lattice model (e.g., 3, 5, 9, 27).
    
    Description:
    - This function initializes the appropriate indices and qubit ranges for each dimension.
      It then applies binary encoding and shifting gates to simulate particle streaming in the lattice model.
    """
    
    # Initialize arrays to hold the indices and qubit information for each dimension
    index_first_qubit_of_dimension = np.zeros(3, dtype=int)
    num_qubits_per_dimension = np.zeros(3, dtype=int)
    num_microscopic_qubits = 0
    lattice_velocities = get_velocity_set(num_velocities)

    # Set up dimension-specific indices and qubit counts based on the velocity model
    if num_velocities == 3:
        num_microscopic_qubits = len(qc.qregs[1])
        index_first_microscopic_qubit = len(qc.qregs[0])
        num_qubits_per_dimension[0] = len(qc.qregs[0])
        index_last_microscopic_qubits = len(qc.qubits) - 1

    elif num_velocities in [5, 9]:
        index_first_qubit_of_dimension[1] = len(qc.qregs[0])
        num_microscopic_qubits = len(qc.qregs[2])
        index_first_microscopic_qubit = len(qc.qregs[0]) + len(qc.qregs[1])
        num_qubits_per_dimension[0] = len(qc.qregs[0])
        num_qubits_per_dimension[1] = len(qc.qregs[1])
        index_last_microscopic_qubits = len(qc.qubits) - 1

    elif num_velocities == 27:
        index_first_qubit_of_dimension[1] = len(qc.qregs[0])
        index_first_qubit_of_dimension[2] = len(qc.qregs[0]) + len(qc.qregs[1])
        num_microscopic_qubits = len(qc.qregs[3])
        index_first_microscopic_qubit = len(qc.qregs[0]) + len(qc.qregs[1]) + len(qc.qregs[2])
        num_qubits_per_dimension[0] = len(qc.qregs[0])
        num_qubits_per_dimension[1] = len(qc.qregs[1])
        num_qubits_per_dimension[2] = len(qc.qregs[2])
        index_last_microscopic_qubits = len(qc.qubits) - 1

    # Iterate through each lattice velocity index to apply streaming operations
    for lattice_index in range(lattice_velocities.shape[1]):
        velocity_vector = lattice_velocities[:, lattice_index]
        
        # Skip the resting velocity as it does not require streaming or encoding
        if all(v == 0 for v in velocity_vector):
            continue

        # Apply binary encoding of the distribution function for the current lattice index
        get_binary_encoding_of_distribution_functions_gate(
            qc, num_microscopic_qubits, lattice_index, index_first_microscopic_qubit
        )

        # Apply right or left shift gates for each dimension based on the velocity vector
        for dimension_index in range(3):
            if velocity_vector[dimension_index] == 1:
                get_right_shift_gate(
                    qc, index_first_qubit_of_dimension, num_qubits_per_dimension, lattice_index,
                    dimension_index, num_microscopic_qubits, index_first_microscopic_qubit, index_last_microscopic_qubits
                )
            elif velocity_vector[dimension_index] == -1:
                get_left_shift_gate(
                    qc, index_first_qubit_of_dimension, num_qubits_per_dimension, lattice_index,
                    dimension_index, num_microscopic_qubits, index_first_microscopic_qubit, index_last_microscopic_qubits
                )

        # Apply binary encoding again after streaming
        get_binary_encoding_of_distribution_functions_gate(
            qc, num_microscopic_qubits, lattice_index, index_first_microscopic_qubit
        )
        
        # Add a barrier for separation of streaming operations for each lattice index
        qc.barrier()

def addition(qc: QuantumCircuit, num_velocities: int) -> None:
    """
    Applies an addition operation in the quantum circuit based on the specified number of velocities.
    This operation involves controlled swaps and Hadamard gates to set up target registers based on
    the source registers in a way that varies with the lattice velocity model.

    Parameters:
    - qc (QuantumCircuit): The quantum circuit where the addition operation will be applied.
    - num_velocities (int): The number of velocities in the lattice model (supported: 3, 5, 9, 27).
    
    Raises:
    - ValueError: If an unsupported number of velocities is passed.
    """
    
    # Set source, target registers, and qubit counts based on velocity model
    if num_velocities == 3:
        source_reg, target_reg = 1, 2
        qubits = 2
    elif num_velocities == 5:
        source_reg, target_reg = 2, 3
        qubits = 3
    elif num_velocities == 9:
        source_reg, target_reg = 2, 3
        qubits = 4
    elif num_velocities == 27:
        source_reg, target_reg = 3, 4
        qubits = 5
    else:
        raise ValueError("Unsupported velocity set: supported values are 3, 5, 9, or 27.")
    
    # Apply swap and Hadamard operations on target register based on source qubits
    for i in range(qubits):
        qc.swap(qc.qregs[target_reg][0], qc.qregs[source_reg][i])
        qc.h(qc.qregs[target_reg][0])  # Apply Hadamard gate to the target qubit after swap



def get_velocity_set(num_velocities: int) -> np.array:
    """
    Returns the lattice velocity set for the specified number of velocities.
    
    Parameters:
    - num_velocities (int): The number of velocities in the lattice model, such as 3, 5, 9, or 27.
    
    Returns:
    - np.array: A 3xN array where each column represents a velocity vector in 1D, 2D, or 3D.
    
    Raises:
    - ValueError: If an unsupported number of velocities is requested.
    """
    
    lattice_velocity_sets = {
        3: np.array([
            [0.0, 1.0, -1.0], 
            [0.0, 0.0, 0.0], 
            [0.0, 0.0, 0.0]
        ]),
        5: np.array([
            [0.0, 1.0, 0.0, -1.0, 0.0], 
            [0.0, 0.0, 1.0, 0.0, -1.0], 
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        9: np.array([
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0], 
            [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0], 
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        27: np.array([
            [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
        ])
    }
    
    try:
        return lattice_velocity_sets[num_velocities]
    except KeyError:
        raise ValueError(f"Unsupported lattice set: {num_velocities}. Supported sets are {list(lattice_velocity_sets.keys())}.")


def QLBM(density_field: np.array , velocity_field: np.array, number_velocities: int) -> QuantumCircuit:
    """
    Constructs a quantum lattice Boltzmann model (QLBM) circuit using specified density and velocity fields.
    
    Parameters:
    - density_field (np.array): Initial density field of the system.
    - velocity_field (np.array): Initial velocity field of the system.
    - number_velocities (int): Number of discrete velocities in the lattice model (e.g., 3, 5, 9, 27).
    
    Returns:
    - QuantumCircuit: The quantum circuit representing the QLBM simulation.
    
    Description:
    - Initializes a quantum circuit with the specified density field.
    - Duplicates the density field in preparation for collision and streaming steps.
    - Applies the collision operation based on the velocity field.
    - Applies the streaming operation to propagate particles across the lattice.
    - Uses addition to update values after streaming.
    """
    
    # Initialize the quantum circuit with the density field and velocity count
    qc = InitializeQC(density_field=density_field, num_velocities=number_velocities)
    
    # Duplicate the density field in preparation for further operations
    duplicate_density_field(qc=qc, num_velocities=number_velocities)
    
    # Apply the collision operator based on the velocity field
    collision(qc=qc, velocity_field=velocity_field, num_velocities=number_velocities)
    
    # Perform streaming to propagate particles across the lattice
    streaming(qc=qc, num_velocities=number_velocities)
    
    # Apply addition to update values after streaming
    addition(qc=qc, num_velocities=number_velocities)
    
    return qc


def QLBMV2(density_field: np.array, velocity_field: np.array, number_velocities: int) -> QuantumCircuit:
    """
    Constructs a quantum lattice Boltzmann model (QLBM) circuit using specified density and velocity fields
    and performs initialization, duplication, collision, and streaming steps.
    
    Parameters:
    - density_field (np.array): Initial density field of the system.
    - velocity_field (np.array): Initial velocity field of the system.
    - number_velocities (int): Number of discrete velocities in the lattice model (e.g., 3, 5, 9, 27).
    
    Returns:
    - QuantumCircuit: The quantum circuit representing the QLBM simulation.
    
    Description:
    - Initializes a quantum circuit with the specified density field.
    - Duplicates the density field in preparation for collision and streaming.
    - Applies the collision operator based on the velocity field.
    - Applies the streaming operation to propagate particles across the lattice.
    - Note: Unlike QLBM, this version omits the addition step. This is done classically as post processing.
    """
    
    # Initialize the quantum circuit with the density field and velocity count
    qc = InitializeQC(density_field=density_field, num_velocities=number_velocities)
    
    # Duplicate the density field in preparation for further operations
    duplicate_density_field(qc=qc, num_velocities=number_velocities)
    
    # Apply the collision operator based on the velocity field
    collision(qc=qc, velocity_field=velocity_field, num_velocities=number_velocities)
    
    # Perform streaming to propagate particles across the lattice
    streaming(qc=qc, num_velocities=number_velocities)
    
    return qc