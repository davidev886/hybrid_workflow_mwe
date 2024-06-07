import numpy as np


def signature_permutation(orbital_list):
    """
    Returns the signature of the permutation in orbital_list
    """
    if len(orbital_list) == 1:
        return True

    transposition_count = 0
    for index, element in enumerate(orbital_list):
        for next_element in orbital_list[index + 1:]:
            if element > next_element:
                transposition_count += 1

    return (-1)**transposition_count


def get_coeff_wf(final_state_vector, n_elec, thres=1e-6):
    """
    :param final_state_vector: State vector from a VQE simulation
    :param n_elec: Number of electrons in active space
    :param ncore_electrons: Number of electrons in core space
    :param thres: Threshold for coefficients to keep from VQE wavefunction
    :returns: Input for ipie trial: coefficients, list of occupied alpha, list of occupied bets
    """
    n_qubits = int(np.log2(final_state_vector.size))
    coeff = []
    occas = []
    occbs = []
    for j, val in enumerate(final_state_vector):
        if abs(val) > thres:
            ket = np.binary_repr(j, width=n_qubits)
            alpha_ket = ket[::2]
            beta_ket = ket[1::2]
            occ_alpha = np.where([int(_) for _ in alpha_ket])[0]
            occ_beta = np.where([int(_) for _ in beta_ket])[0]
            occ_orbitals = np.append(2 * occ_alpha, 2 * occ_beta + 1)

            if (len(occ_alpha) == n_elec[0]) and (len(occ_beta) == n_elec[1]):
                coeff.append(signature_permutation(occ_orbitals) * val)
                occas.append(occ_alpha)
                occbs.append(occ_beta)

    coeff = np.array(coeff, dtype=complex)
    ixs = np.argsort(np.abs(coeff))[::-1]
    coeff = coeff[ixs]
    occas = np.array(occas)[ixs]
    occbs = np.array(occbs)[ixs]

    return coeff, occas, occbs
