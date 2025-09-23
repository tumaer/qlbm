from matplotlib.pylab import qr
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import UCRYGate
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile




def get_right_shift_gate(qc,index_first_qubit_of_dimension,num_qubits_per_dimension):

    qubit_list_to_shift = [
        *range(
            index_first_qubit_of_dimension,
            index_first_qubit_of_dimension + num_qubits_per_dimension,
        )
    ]

    for ii in range(qubit_list_to_shift[-1], qubit_list_to_shift[0], -1):
        control_qubits = [*range(qubit_list_to_shift[0], ii)]
        qc.mcx(control_qubits, ii)
    qc.x(qubit_list_to_shift[0])

def get_left_shift_gate(qc,index_first_qubit_of_dimension,num_qubits_per_dimension):
    qubit_list_to_shift = [
        *range(
            index_first_qubit_of_dimension,
            index_first_qubit_of_dimension + num_qubits_per_dimension,
        )
    ]
    qc.x(qubit_list_to_shift[0])
    for ii in range(qubit_list_to_shift[0]+1 , qubit_list_to_shift[-1] + 1, 1):
        control_qubits = [*range(qubit_list_to_shift[0], ii)]
        qc.mcx(control_qubits, ii)
        
def get_binary_encoding_of_distribution_functions_gate(qc,num_qubits,lattice_index,index_first_qubit):
    lattice_index_as_binary_list = [int(d) for d in str(format(lattice_index, f"0{num_qubits}b"))]
    lattice_index_as_binary_list.reverse()

    for i, binary_value in enumerate(lattice_index_as_binary_list):
        if binary_value == 0:
            qc.x(index_first_qubit + i)

def get_collision_non_uniform_velocity(qc,theta_collision):
    qc.barrier()
    qubit_ry_list = list(range(len(qc.qubits)))
    qc.append(UCRYGate(theta_collision.tolist()),qubit_ry_list)


def computecollisionangle(microscopic_velocities: int, u_LBM: float):
    c_s = 1/np.sqrt(3)

    if(microscopic_velocities==3):

        collision_angle = np.sqrt(0.5+0.5*u_LBM/c_s**2)
        theta_collision= 2*np.arccos(collision_angle)

    elif(microscopic_velocities==9):
        u_flattened = u_LBM[:,:,0].flatten(order='C')
        v_flattened = u_LBM[:,:,1].flatten(order='C')

        theta_collision_angle = np.zeros([np.size(u_flattened),4])
        theta_collision_angle[:,0] = np.sqrt(0.5*(1+u_flattened/(c_s**2)))
        theta_collision_angle[:,1] = np.sqrt(0.5*(1+v_flattened/(c_s**2)))
        theta_collision_angle[:,2] = np.sqrt(0.5*(1+(u_flattened+v_flattened)/(c_s**2)))
        theta_collision_angle[:,3] = np.sqrt(0.5*(1+(-u_flattened+v_flattened)/(c_s**2)))


        theta_collision = np.zeros([np.size(u_flattened),4])
        theta_collision[:,0] = 2*np.arccos(theta_collision_angle[:,0])
        theta_collision[:,1] = 2*np.arccos(theta_collision_angle[:,1])
        theta_collision[:,2] = 2*np.arccos(theta_collision_angle[:,2])
        theta_collision[:,3] = 2*np.arccos(theta_collision_angle[:,3])

    else:
        raise ValueError("microscopic_velocities must be either 3 or 9.")

    return theta_collision
    
def computeconstantangles(microscopic_velocities: int):
    if (microscopic_velocities==3):
        theta_weight = 2*np.arccos(np.sqrt(2/3))
    elif (microscopic_velocities==9):
        # between f_0 and rest
        theta_weight = np.zeros(4)
        theta_weight[0] = 2*np.arccos(np.sqrt(4/9))
        theta_weight[1] = 2*np.arccos(np.sqrt(2/5))
        theta_weight[2] = 2*np.arccos(np.sqrt(2/3)) 
        theta_weight[3] = 2*np.arccos(np.sqrt(1/2))
    else:
        raise ValueError("microscopic_velocities must be either 3 or 9.")
    return theta_weight



def one_time_step(qc: QuantumCircuit, theta_weight: float, theta_collision: float, microscopic_velocities: int):

    # One time step D1Q3
    if (microscopic_velocities==3):
        #get quantum registers
        qq = qc.qregs[0]
        qx = qc.qregs[1]
        sol = qc.cregs[0]
        #first qubit is ancilla
        index_of_first_qubit_of_dimension = 1
        num_qubits_per_dimension = np.size(qx)
        qc.barrier()
        qc.ry(theta_weight,qq)
        qc.measure(qq,sol[-1])
        with qc.if_test((sol[-1],1)):
            qc.reset(qq)
            get_collision_non_uniform_velocity(qc,theta_collision=theta_collision)   
            qc.barrier()
            qc.measure(qq,sol[-1])
            with qc.if_test((sol[-1],0)) as else_:
                #streaming positive
                get_right_shift_gate(qc,index_first_qubit_of_dimension=index_of_first_qubit_of_dimension,num_qubits_per_dimension=num_qubits_per_dimension)
            with else_:
                qc.reset(qq)
                #streaming negative
                get_left_shift_gate(qc,index_first_qubit_of_dimension=index_of_first_qubit_of_dimension,num_qubits_per_dimension=num_qubits_per_dimension)

    # One time step D2Q9
    if (microscopic_velocities==9):
        #get quantum registers
        qq = qc.qregs[0]
        qx = qc.qregs[1]
        qy = qc.qregs[2]
        sol = qc.cregs[0]

        index_first_qubit_of_dimension=1
        number_x_qubits = np.size(qx)
        number_y_qubits = np.size(qy)
        qc.barrier()
        #f_0 or rest
        qc.ry(theta_weight[0],qq)
        qc.measure(qq,sol[-1])
        with qc.if_test((sol[-1],1)):
            qc.reset(qq)
            #f_1,2,3,4 or rest
            qc.ry(theta_weight[1],qq)
            qc.measure(qq,sol[-1])
            #its now f_1,2
            with qc.if_test((sol[-1],0)) as else_0:
                get_collision_non_uniform_velocity(qc,theta_collision[:,0])
                qc.measure(qq,sol[-1])
                #now its f_1
                with qc.if_test((sol[-1],0)) as else_1:
                    get_right_shift_gate(qc,index_first_qubit_of_dimension,number_x_qubits)
                #now its f_2
                with else_1:
                    qc.reset(qq)
                    get_left_shift_gate(qc,index_first_qubit_of_dimension,number_x_qubits)
            # now its f_3-8
            with else_0:
                qc.reset(qq)
                qc.ry(theta_weight[2],qq)
                qc.measure(qq,sol[-1])
                #now its f_3,4
                with qc.if_test((sol[-1],0)) as else_1:
                    get_collision_non_uniform_velocity(qc,theta_collision[:,1])
                    qc.measure(qq,sol[-1])
                    # now its f_3
                    with qc.if_test((sol[-1],0)) as else_2:
                        get_right_shift_gate(qc,index_first_qubit_of_dimension+number_x_qubits,number_y_qubits)
                    # now its f_4
                    with else_2:
                        qc.reset(qq)
                        get_left_shift_gate(qc,index_first_qubit_of_dimension+number_x_qubits,number_y_qubits)
            
                #now its f_5-8
                with else_1:
                    qc.reset(qq)
                    qc.ry(theta_weight[3],qq)
                    qc.measure(qq,sol[-1])
                    #now its f_5,7
                    with qc.if_test((sol[-1],0)) as else_2:
                        get_collision_non_uniform_velocity(qc,theta_collision[:,2])
                        qc.measure(qq,sol[-1])
                        #now its f_5
                        with qc.if_test((sol[-1],0)) as else_3:
                            get_right_shift_gate(qc,index_first_qubit_of_dimension,number_x_qubits)
                            get_right_shift_gate(qc,index_first_qubit_of_dimension+number_x_qubits,number_y_qubits)
                        #now its f_7
                        with else_3:
                            qc.reset(qq)
                            get_left_shift_gate(qc,index_first_qubit_of_dimension,number_x_qubits)
                            get_left_shift_gate(qc,index_first_qubit_of_dimension+number_x_qubits,number_y_qubits)
                    # now its f_6,8
                    with else_2:
                        qc.reset(qq)
                        get_collision_non_uniform_velocity(qc,theta_collision[:,3])
                        qc.measure(qq,sol[-1])
                        #now its f_6
                        with qc.if_test((sol[-1],0)) as else_3:
                            get_left_shift_gate(qc,index_first_qubit_of_dimension,number_x_qubits)
                            get_right_shift_gate(qc,index_first_qubit_of_dimension+number_x_qubits,number_y_qubits)
                        #now its f_8
                        with else_3:
                            qc.reset(qq)
                            get_right_shift_gate(qc,index_first_qubit_of_dimension,number_x_qubits)
                            get_left_shift_gate(qc,index_first_qubit_of_dimension+number_x_qubits,number_y_qubits)



def measurement_1D(qc: QuantumCircuit):
    qq = qc.qregs[0]
    sol = qc.cregs[0]
    qx = qc.qregs[1]
    number_x_qubits = np.size(qx)

    qc.reset(qq)
    qc.measure(qq,sol[-1])
    qc.measure(np.arange(1,number_x_qubits+1),sol[:number_x_qubits])

def measurement_2D(qc: QuantumCircuit):
    qq = qc.qregs[0]
    sol = qc.cregs[0]
    qx = qc.qregs[1]
    qy = qc.qregs[2]
    number_x_qubits = np.size(qx)
    number_y_qubits = np.size(qy)

    qc.reset(qq)
    qc.measure(np.arange(1,number_x_qubits+number_y_qubits+1),sol[:number_x_qubits+number_y_qubits])
    qc.measure(qq,sol[-1])


def is_power_of_two(n):
    return (n > 0) and (n & (n - 1)) == 0

def initialize(density):
    #check if shape is correct
    if all(is_power_of_two(dim) for dim in density.shape):
        pass
    else:
        raise ValueError(f"Shape mismatch! All dimensions must be powers of two, but got {density.shape}.")
    

    dimension = density.ndim
    uniform = np.all(density == density[0])

    if dimension == 1:
        number_of_entries_x = density.shape[0]
        microscopic_velocities = 3
        density_flattened = density.flatten(order='C')
        density_sqrt = np.sqrt(density_flattened)
        normalization_constant = np.linalg.norm(density_sqrt)
        number_qx= np.log2(number_of_entries_x)
        qx = QuantumRegister(int(number_qx),name='x')
        qq = QuantumRegister(1,name='q')
        sol = ClassicalRegister(int(number_qx+1),name='sol')
        qc = QuantumCircuit(qq,qx,sol)
        if uniform:
            qc.h(qx)
        else:
            density_sqrt_normalized = density_sqrt/normalization_constant
            qc.initialize(density_sqrt_normalized,qx)



    elif dimension == 2:
        microscopic_velocities = 9
        number_of_entries_y = density.shape[0]
        number_of_entries_x = density.shape[1]
        density_flattened = density.flatten(order='C')
        density_sqrt = np.sqrt(density_flattened)
        normalization_constant = np.linalg.norm(density_sqrt)
        number_qx = np.log2(number_of_entries_x)
        number_qy = np.log2(number_of_entries_y)
        qx = QuantumRegister(int(number_qx),name='x')
        qy = QuantumRegister(int(number_qy),name='y')
        qq = QuantumRegister(1,name='q')
        sol = ClassicalRegister(int(number_qx+number_qy+1),name='sol')
        qc = QuantumCircuit(qq,qx,qy,sol)
        if uniform:
            qc.h(qx[:]+qy[:])
        else:
            density_sqrt_normalized = density_sqrt/normalization_constant
            qc.initialize(density_sqrt_normalized,qx[:]+qy[:])


    return qc, normalization_constant


def run_circuit(qc,shots,normalization_constant, recreate_paper_results=False):
    if(recreate_paper_results):
        #Seed to reproduce results from paper
        SEED = 1234567
    sim_no_noise = AerSimulator()

    sim_no_noise.set_options(
    max_parallel_threads = 0,
    max_parallel_experiments = 0,
    max_parallel_shots = 16,
    statevector_parallel_threshold = 16
    )


    circ = transpile(qc, sim_no_noise)
    if recreate_paper_results:
        result = sim_no_noise.run(circ,shots=shots,dynamic=True,seed_simulator=SEED).result()
    else:
        result = sim_no_noise.run(circ,shots=shots,dynamic=True).result()
    counts = result.get_counts()
    probabilities = {state: count / shots for state, count in counts.items()}
    sorted_states = sorted(probabilities.keys())
    sorted_probabilities = [probabilities[state] for state in sorted_states]
    result= np.array(sorted_probabilities)*np.linalg.norm(normalization_constant)**2

    return result


def create_QLBM_circuit(density, advection_velocity, timesteps):

    if density.ndim == 1:
        microscopic_velocities = 3
    elif density.ndim == 2:
        microscopic_velocities = 9
    else:
        raise ValueError("Density must be either 1D or 2D.")
    
    qc, normalization_constant = initialize(density)
    for _ in range(timesteps):
        one_time_step(qc, computeconstantangles(microscopic_velocities), computecollisionangle(microscopic_velocities, advection_velocity), microscopic_velocities)

    if density.ndim == 1:
        measurement_1D(qc)
    elif density.ndim == 2:
        measurement_2D(qc)

    return qc, normalization_constant
    


#### Code for Hybrid approach ####

def create_hybrid_circuits(density, theta_collision, microscopic_velocities, sequence):
    qcs = []

    if density.ndim == 1:
        for shot in range(sequence.shape[0]):
            qc, normalization_constant = initialize(density)
            qq = qc.qregs[0]
            qx = qc.qregs[1]
            sol = qc.cregs[0]
            #first qubit is ancilla
            index_of_first_qubit_of_dimension = 1
            num_qubits_per_dimension = np.size(qx)
            for timestep in range(sequence.shape[1]):
                # if == 0 do nothing
                if sequence[shot, timestep] == 1:
                    get_collision_non_uniform_velocity(qc,theta_collision=theta_collision)   
                    qc.barrier()
                    qc.measure(qq,sol[-1])
                    with qc.if_test((sol[-1],0)) as else_:
                        #streaming positive
                        get_right_shift_gate(qc,index_first_qubit_of_dimension=index_of_first_qubit_of_dimension,num_qubits_per_dimension=num_qubits_per_dimension)
                    with else_:
                        qc.reset(qq)
                        #streaming negative
                        get_left_shift_gate(qc,index_first_qubit_of_dimension=index_of_first_qubit_of_dimension,num_qubits_per_dimension=num_qubits_per_dimension) 
            measurement_1D(qc)

            qcs.append(qc)

    else:
        raise ValueError("Currently only 1D implemented for hybrid approach.")

    return qcs, normalization_constant


def hybrid_classical_quantum_LBM(density, advection_velocity, timesteps, shots,  recreate_paper_results=False):
    if recreate_paper_results:
        #Seed to reproduce results from paper
        SEED = 1234567
        rng = np.random.default_rng(SEED)
    else:
        rng = np.random.default_rng()

    if density.ndim == 1:
        microscopic_velocities = 3
        #p= [w_0, w_1+w_2]
        sequence = rng.choice([0, 1], size=(shots,timesteps), p=[2/3, 1/3])
    elif density.ndim == 2:
        #p = [w_0, w_1+w_2, w_3+w_4, w_5+w_7, w_6+w_8]
        microscopic_velocities = 9
        sequence = rng.choice([0, 1, 2, 3, 4], size=(shots,timesteps), p=[4/9, 2/9, 2/9, 2/36, 2/36])
    else:
        raise ValueError("Density must be either 1D or 2D.")


    collision_angles = computecollisionangle(microscopic_velocities, advection_velocity)

    qcs, normalization_constant = create_hybrid_circuits(density, collision_angles, microscopic_velocities, sequence)

    return qcs, normalization_constant

def run_hybrid_classical_quantum_LBM(qcs,normalization_constant):
    sim_no_noise = AerSimulator()

    sim_no_noise.set_options(
    max_parallel_threads = 0,
    max_parallel_experiments = 0,
    max_parallel_shots = 16,
    statevector_parallel_threshold = 16
    )

    total_qubits = len(qcs[0].qubits)
    possible_outcomes = {format(i, f'0{total_qubits}b'): 0 for i in range(2 ** total_qubits)}

    shots = 1
    circs = transpile(qcs, sim_no_noise, optimization_level=0)
    for ii in range(len(circs)):
        result = sim_no_noise.run(circs[ii],shots=shots).result()

        counts = result.get_counts()
        outcome = list(counts.keys())[0]
        possible_outcomes[outcome] +=1


    probabilities = {state: possible_outcomes / len(circs) for state, possible_outcomes in possible_outcomes.items()}
    sorted_states = sorted(probabilities.keys())
    sorted_probabilities = [probabilities[state] for state in sorted_states]
    solution_quantum = np.array(sorted_probabilities[:2**(total_qubits-1)])*np.linalg.norm(normalization_constant)**2

    return solution_quantum