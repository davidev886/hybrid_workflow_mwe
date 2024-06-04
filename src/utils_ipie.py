import numpy as np


def normal_ordering_swap(l):
    """
    This function normal calculates the phase coefficient (-1)^swaps, where swaps is the number of
    swaps required to normal order a given list of numbers.
    :param l: list of numbers, e.g. orbitals
    :returns: number of required swaps
    """
    count = 0
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i] > l[j]:
                count += 1

    return count


def convert_state_big_endian(state_little_endian):
    """
    Convert cudaq state vector from little endian to big endian notation
    """
    state_big_endian = 0. * state_little_endian

    n_qubits = int(np.log2(state_big_endian.size))
    for j, val in enumerate(state_little_endian):
        little_endian_pos = np.binary_repr(j, n_qubits)
        big_endian_pos = little_endian_pos[::-1]
        int_big_endian_pos = int(big_endian_pos, 2)
        state_big_endian[int_big_endian_pos] = state_little_endian[j]

    return state_big_endian


def get_coeff_wf(final_state_vector, n_elec, ncore_electrons=0, thres=1e-6):
    """
    :param final_state_vector: State vector from a VQE simulation
    :param n_elec: Number of electrons in active space
    :param ncore_electrons: Number of electrons in core space
    :param thres: Threshold for coefficients to keep from VQE wavefunction
    :returns: Input for ipie trial: coefficients, list of occupied alpha, list of occupied bets
    """
    bin_ind = [np.binary_repr(i, width=int(np.log2(len(final_state_vector)))) for i in
               range(len(final_state_vector))]
    coeff = []
    occas = []
    occbs = []

    for k, i in enumerate(bin_ind):
        alpha_aux = []
        beta_aux = []
        for j in range(len(i) // 2):
            alpha_aux.append(i[2 * j])
            beta_aux.append(i[2 * j + 1])
        alpha_occ = [i for i, x in enumerate(alpha_aux) if x == '1']
        beta_occ = [i for i, x in enumerate(beta_aux) if x == '1']

        if (np.abs(final_state_vector[k]) >= thres) and (len(alpha_occ) == n_elec[0]) and (len(beta_occ) == n_elec[1]):
            coeff.append(final_state_vector[k])
            occas.append(alpha_occ)
            occbs.append(beta_occ)
    # We need it non_normal ordered
    for i in range(len(coeff)):
        coeff[i] = (-1) ** (
            normal_ordering_swap([2 * j for j in occas[i]] + [2 * j + 1 for j in occbs[i]])) * \
                   coeff[i]

    core = [i for i in range(ncore_electrons)]
    occas = [np.array(core + [o + ncore_electrons for o in oa]) for oa in occas]
    occbs = [np.array(core + [o + ncore_electrons for o in ob]) for ob in occbs]

    coeff = np.array(coeff, dtype=complex)
    ixs = np.argsort(np.abs(coeff))[::-1]
    coeff = coeff[ixs]
    occas = np.array(occas)[ixs]
    occbs = np.array(occbs)[ixs]

    return coeff, occas, occbs
