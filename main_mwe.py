"""
Contains the main file for running a complete VQE + AFQMC
"""

import os
import numpy as np
from pyscf import gto, scf, ao2mo, mcscf
from openfermion.transforms import jordan_wigner
from openfermion import generate_hamiltonian
import time
import h5py


# from pyscf import gto, scf, fci, ao2mo, mcscf, lib
# import sys
# import pickle
# import json
# from pyscf.lib import chkfile
# from pyscf.scf.chkfile import dump_scf
# from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
# import shutil
# from openfermion.linalg import get_sparse_operator
# from src.spin_square import of_spin_operator
# from openfermion.hamiltonians import s_squared_operator

# from collections import defaultdict
# import pandas as pd
# import pickle
# import json

if __name__ == "__main__":
    np.random.seed(12)
    target = "nvidia"
    optimizer_type = "cudaq"

    do_vqe = False
    do_afqmc = True
    num_active_orbitals = 6
    num_active_electrons = 8

    spin = 0
    charge = 0
    nocca_act = (num_active_electrons + spin) // 2
    noccb_act = (num_active_electrons - spin) // 2

    atom = "systems/O2_spin_0/geo.xyz"
    basis = "cc-pVQZ"
    ipie_input_dir = "test_mwe"
    chkptfile_rohf = "scf_mwe.chk"
    file_chk = os.path.join(ipie_input_dir, chkptfile_rohf)
    os.makedirs(ipie_input_dir, exist_ok=True)

    n_vqe_layers = 1

    mol = gto.M(
        atom=atom,
        spin=spin,
        charge=charge,
        basis=basis,
        verbose=4
    )
    nocca, noccb = mol.nelec
    mol_nelec = mol.nelec
    mf = scf.ROHF(mol)
    print("# saving chkfile to", os.path.join(ipie_input_dir, chkptfile_rohf))
    mf.chkfile = file_chk
    mf.kernel()

    if do_vqe:
        from src.vqe_cudaq_qnp import VqeQnp
        from src.utils_cudaq import get_cudaq_hamiltonian

        my_casci = mcscf.CASCI(mf, num_active_orbitals, num_active_electrons)
        ss = (mol.spin / 2 * (mol.spin / 2 + 1))
        my_casci.fix_spin_(ss=ss)

        e_tot, e_cas, fcivec, mo_output, mo_energy = my_casci.kernel()

        print('# FCI Energy in CAS:', e_tot)

        h1, energy_core = my_casci.get_h1eff()
        h2 = my_casci.get_h2eff()
        h2_no_symmetry = ao2mo.restore('1', h2, num_active_orbitals)
        tbi = np.asarray(h2_no_symmetry.transpose(0, 2, 3, 1), order='C')

        mol_ham = generate_hamiltonian(h1, tbi, energy_core.item(), EQ_TOLERANCE=1e-8)
        jw_hamiltonian = jordan_wigner(mol_ham)

        print("# Target", target)

        start = time.time()
        hamiltonian_cudaq, energy_core = get_cudaq_hamiltonian(jw_hamiltonian)
        end = time.time()
        print("# Time for preparing the cudaq hamiltonian:", end - start)

        n_qubits = 2 * num_active_orbitals

        n_alpha_vec = np.array([1] * nocca_act + [0] * (num_active_orbitals - nocca_act))
        n_beta_vec = np.array([1] * noccb_act + [0] * (num_active_orbitals - noccb_act))
        init_mo_occ = (n_alpha_vec + n_beta_vec).tolist()

        options = {'maxiter': 50000,
                   'optimizer_type': optimizer_type,
                   'energy_core': energy_core,
                   'initial_parameters': None,
                   'return_final_state_vec': True}

        print("# init_mo_occ", init_mo_occ)
        print("# layers", n_vqe_layers)

        time_start = time.time()

        vqe = VqeQnp(n_qubits=n_qubits,
                     n_layers=n_vqe_layers,
                     init_mo_occ=init_mo_occ,
                     target=target)

        results = vqe.run_vqe_cudaq(hamiltonian_cudaq, options=options)
        energy_optimized = results['energy_optimized']

        time_end = time.time()
        print(f"# Best energy {energy_optimized}")
        print(f"# VQE time {time_end - time_start}")
        print(results["state_vec"])
        final_state_vector = results["state_vec"]
    else:
        final_state_vector = np.loadtxt("o2_wf.dat", dtype=np.complex_)

    if do_afqmc:
        # import cupy
        from ipie.config import config
        config.update_option("use_gpu", True)

        from ipie.hamiltonians.generic import Generic as HamGeneric
        from ipie.qmc.afqmc import AFQMC
        from ipie.systems.generic import Generic
        from ipie.trial_wavefunction.particle_hole import ParticleHole
        from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk

        from src.input_ipie import get_coeff_wf

        coeff, occas, occbs = get_coeff_wf(final_state_vector,
                                           ncore_electrons=0,
                                           n_elec=(nocca_act, noccb_act)
                                           )

        # Need to write wavefunction to checkpoint file.
        with h5py.File(file_chk, "r+") as fh5:
            fh5["mcscf/ci_coeffs"] = coeff
            fh5["mcscf/occs_alpha"] = occas
            fh5["mcscf/occs_beta"] = occbs

        # generate input file for ipie
        from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
        gen_ipie_input_from_pyscf_chk(file_chk,
                                      mcscf=True,
                                      chol_cut=1e-5)

        with h5py.File("hamiltonian.h5") as fa:
            chol = fa["LXmn"][()]
            h1e = fa["hcore"][()]
            e0 = fa["e0"][()]

        num_basis = chol.shape[1]

        system = Generic(nelec=mol_nelec)

        num_chol = chol.shape[0]
        ham = HamGeneric(
            np.array([h1e, h1e]),
            chol.transpose((1, 2, 0)).reshape((num_basis * num_basis, num_chol)),
            e0,
        )

        # Build trial wavefunction
        with h5py.File("wavefunction.h5", "r") as fh5:
            coeff = fh5["ci_coeffs"][:]
            occa = fh5["occ_alpha"][:]
            occb = fh5["occ_beta"][:]

        wavefunction = (coeff, occa, occb)
        trial = ParticleHole(
            wavefunction,
            mol_nelec,
            num_basis,
            num_dets_for_props=len(wavefunction[0]),
            verbose=True,
        )
        trial.compute_trial_energy = True
        trial.build()
        trial.half_rotate(ham)

        afqmc_msd = AFQMC.build(
            mol_nelec,
            ham,
            trial,
            num_walkers=100,
            num_steps_per_block=25,
            num_blocks=10,
            timestep=0.005,
            stabilize_freq=5,
            seed=96264512,
            pop_control_freq=5,
            verbose=True,
        )
        afqmc_msd.run()
        afqmc_msd.finalise(verbose=True)
