"""
    Contains the class with the VQE using the quantum-number-preserving ansatz
"""
import numpy as np

import cudaq
from src.utils_cudaq import buildOperatorMatrix
import pandas as pd


class VqeQnp(object):
    """
        Implements the quantum-number-preserving ansatz proposed by Anselmetti et al. NJP 23 (2021)
    """

    def __init__(self,
                 n_qubits,
                 n_layers,
                 init_mo_occ=None,
                 target="nvidia",
                 system_name="FeNTA"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.number_of_Q_blocks = n_qubits // 2 - 1
        self.num_params = 2 * self.number_of_Q_blocks * n_layers
        self.init_mo_occ = init_mo_occ
        self.final_state_vector_best = None
        self.best_vqe_params = None
        self.best_vqe_energy = None
        self.target = target
        self.initial_x_gates_pos = self.prepare_initial_circuit()

        self.spin_s_square = buildOperatorMatrix("total", n_qubits)
        self.spin_s_z = buildOperatorMatrix("projected", n_qubits)
        self.system_name = system_name

    def prepare_initial_circuit(self):
        """
        Creates a list with the position of the X gates that should be applied to the initial |00...0>
        state to set the number of electrons and the spin correctly
        """
        x_gates_pos_list = []
        if self.init_mo_occ is not None:
            for idx_occ, occ in enumerate(self.init_mo_occ):
                if int(occ) == 2:
                    x_gates_pos_list.extend([2 * idx_occ, 2 * idx_occ + 1])
                elif int(occ) == 1:
                    x_gates_pos_list.append(2 * idx_occ)

        return x_gates_pos_list

    def layers(self):
        """
            Generates the QNP ansatz circuit and returns the  kernel and the optimization paramenters thetas

            params: list/np.array
            [theta_0, ..., theta_{M-1}, phi_0, ..., phi_{M-1}]
            where M is the total number of blocks = layer * (n_qubits/2 - 1)

            returns: kernel
                     thetas
        """
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        number_of_blocks = self.number_of_Q_blocks

        cudaq.set_target(self.target)  # nvidia or nvidia-mgpu

        kernel, thetas = cudaq.make_kernel(list)
        # Allocate n qubits.
        qubits = kernel.qalloc(n_qubits)

        for init_gate_position in self.initial_x_gates_pos:
            kernel.x(qubits[init_gate_position])

        count_params = 0
        for idx_layer in range(n_layers):
            for starting_block_num in [0, 1]:
                for idx_block in range(starting_block_num, number_of_blocks, 2):
                    qubit_list = [qubits[2 * idx_block + j] for j in range(4)]

                    # PX gates decomposed in terms of standard gates
                    # and NO controlled Y rotations.
                    # See Appendix E1 of Anselmetti et al New J. Phys. 23 (2021) 113010

                    a, b, c, d = qubit_list
                    kernel.cx(d, b)
                    kernel.cx(d, a)
                    kernel.rz(parameter=-np.pi / 2, target=a)
                    kernel.s(b)
                    kernel.h(d)
                    kernel.cx(d, c)
                    kernel.cx(b, a)
                    kernel.ry(parameter=(1 / 8) * thetas[count_params], target=c)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=d)
                    kernel.rz(parameter=+np.pi / 2, target=a)
                    kernel.cz(a, d)
                    kernel.cx(a, c)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=d)
                    kernel.ry(parameter=(+1 / 8) * thetas[count_params], target=c)
                    kernel.cx(b, c)
                    kernel.cx(b, d)
                    kernel.rz(parameter=+np.pi / 2, target=b)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=c)
                    kernel.ry(parameter=(+1 / 8) * thetas[count_params], target=d)
                    kernel.cx(a, c)
                    kernel.cz(a, d)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params], target=c)
                    kernel.ry(parameter=(1 / 8) * thetas[count_params], target=d)
                    kernel.cx(d, c)
                    kernel.h(d)
                    kernel.cx(d, b)
                    kernel.s(d)
                    kernel.rz(parameter=-np.pi / 2, target=b)
                    kernel.cx(b, a)
                    count_params += 1

                    # Orbital rotation
                    kernel.fermionic_swap(np.pi, b, c)
                    kernel.givens_rotation((-1 / 2) * thetas[count_params], a, b)
                    kernel.givens_rotation((-1 / 2) * thetas[count_params], c, d)
                    kernel.fermionic_swap(np.pi, b, c)
                    count_params += 1

        return kernel, thetas

    def get_state_vector(self, param_list):
        """
        Returns the state vector generated by the ansatz with paramters given by param_list
        """
        kernel, thetas = self.layers()
        state = convert_state_big_endian(np.array(cudaq.get_state(kernel, param_list), dtype=complex))
        return state

    def get_coeff_wf_ipie(self, param_list):
        """
            :returns: Input for ipie trial: coefficients, list of occupied alpha, list of occupied bets
        """
        kernel, thetas = self.layers()
        state = convert_state_big_endian(np.array(cudaq.get_state(kernel, param_list), dtype=complex))
        return state

    def run_vqe_cudaq(self, hamiltonian, options=None):
        """
        Run VQE
        """
        mpi_support = options.get("mpi_support", False)
        return_final_state_vec = options.get("return_final_state_vec", False)
        if mpi_support:
            cudaq.mpi.initialize()
            print('# mpi is initialized? ', cudaq.mpi.is_initialized())
            num_ranks = cudaq.mpi.num_ranks()
            rank = cudaq.mpi.rank()
            print('# rank', rank, 'num_ranks', num_ranks)

        optimizer = cudaq.optimizers.COBYLA()
        if options['initial_parameters']:
            optimizer.initial_parameters = np.random.rand(self.num_params)
        else:
            optimizer.initial_parameters = np.random.rand(self.num_params)

        kernel, thetas = self.layers()
        maxiter = options.get('maxiter', 100)
        optimizer.max_iterations = options.get('maxiter', maxiter)
        optimizer_type = "cudaq"
        callback_energies = []

        def eval(theta):
            """
            Compute the energy by cudaq.observe
            """
            exp_val = cudaq.observe(kernel,
                                    hamiltonian,
                                    theta).expectation()

            callback_energies.append(exp_val)
            return exp_val

        if optimizer_type == "cudaq":
            print("# Using cudaq optimizer")
            energy_optimized, best_parameters = optimizer.optimize(self.num_params, eval)

            # We add here the energy core
            energy_core = options.get('energy_core', 0.)
            total_opt_energy = energy_optimized + energy_core
            callback_energies = [en + energy_core for en in callback_energies]

            info_final_state = dict()
            print("# Num Params:", self.num_params)
            print("# Qubits:", self.n_qubits)
            print("# N_layers:", self.n_layers)
            print("# Energy after the VQE:", total_opt_energy)

            info_final_state["total_opt_energy"] = total_opt_energy

            df = pd.DataFrame(info_final_state, index=[0])
            df.to_csv(f'{self.system_name}_info_final_state_{self.n_layers}_layers_opt_{optimizer_type}.csv',
                      index=False)

            result = {"energy_optimized": total_opt_energy,
                      "best_parameters": best_parameters,
                      "callback_energies": callback_energies}

            if return_final_state_vec:
                result["state_vec"] = self.get_state_vector(best_parameters)
            return result

        else:
            print(f"# Optimizer {optimizer_type} not implemented")
            exit()

    def compute_energy(self, hamiltonian, params):
        """
            For checking the energy at the end - not used for the VQE
        """
        kernel, thetas = self.layers()

        exp_val = cudaq.observe(kernel,
                                hamiltonian,
                                params).expectation()
        print("# Parameters", params)
        print("# Energy from vqe", exp_val)
        return exp_val


def convert_state_big_endian(state_little_endian):

    state_big_endian = 0. * state_little_endian

    n_qubits = int(np.log2(state_big_endian.size))
    for j, val in enumerate(state_little_endian):
        little_endian_pos = np.binary_repr(j, n_qubits)
        big_endian_pos = little_endian_pos[::-1]
        int_big_endian_pos = int(big_endian_pos, 2)
        state_big_endian[int_big_endian_pos] = state_little_endian[j]

    return state_big_endian
