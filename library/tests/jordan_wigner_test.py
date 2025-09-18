from sys import path
path.append('./')  # To ensure the library module is found when running the test directly

from library.utils.jordan_wigner import (
    FH_zero_particle_state,
    c,
    c_dag,
    n_op,
    full_n_op,
    density_matrix_size_bytes,
    embed,
    emulated_FH_Hamiltonian,
    FH_energy_shift
)
from feedback_grape.utils.operators import (
    sigmaz,
    sigmap,
    sigmam,
    identity,
)
from jax import numpy as jnp

"""
Tests for the Jordan-Wigner transformation functions.
Tests include:
1. Anticommutation relations of creation and annihilation operators.
2. Equivalence of the Heisenberg Hamiltonian and the Fermi-Hubbard Hamiltonian.
3. Correctness of the initial state in terms of particle number.
4. Correctness of creation and annihilation operators on the initial state.
"""

# Initialize spin J = 0, 0.5 Fermi Hubbard model on n sites
n  = 3 # e^- sites
J  = 0.    # Spin
mu = 1.0    # Chemical potential
U  = 2.0    # On-site interaction
t  = 3.0    # Transfer parameter

# Parameters for Heisenberg model
gx = -t
gz = U/4*int(J != 0) # On-site interaction only for spin 1/2 fermions
eps = 2*gz - mu
N_qubits = int(n*(2*J+1))
boundary = 'open'  # 'open' for 1D chain or 'periodic' for 1D ring

if density_matrix_size_bytes(n, J) > 1e9:  # Only run test if density matrix is small enough
    raise ValueError("Density matrix is too large to handle.")

if   J == 0.0: spins = [0]
elif J == 0.5: spins = [-0.5, 0.5]
else: raise ValueError("Invalid spin value for given J.")

if boundary not in ['open', 'periodic']:
    raise ValueError("Invalid boundary condition.")

# Initialize Heisenberg state |down,down,down,...> and Hamiltonian
psi = FH_zero_particle_state(n, J)

H_QS = (
    +0.5*eps*sum(embed(sigmaz(), j, s, n, J) for j in range(n) for s in spins)
    +gx*sum(embed(sigmap(), j, s, n, J)@embed(sigmam(), (j+1)%n, s, n, J) + embed(sigmap(), (j+1)%n, s, n, J)@embed(sigmam(), j, s, n, J) for j in range(n-(boundary=='open')) for s in spins)
)

if J != 0.0: # Onsite interaction not for spinless system
    H_QS += +gz*sum(embed(sigmaz(), j, 0.5, n, J)@embed(sigmaz(), j, -0.5, n, J) for j in range(n))

Delta_E = FH_energy_shift(n, J, mu, U)  # Constant energy shift not mentioned in literature
H_QS += Delta_E*identity(2**N_qubits)

# Initialize Fermi-Hubbard Hamiltonian
H_FH = emulated_FH_Hamiltonian(n, J, mu, t, U, boundary)

# Run tests

# Test 1: Check anticommutation relations
print("Running Test 1: Anticommutation relations...")
for j1 in range(n):
    for j2 in range(n):
        for s1 in spins:
            for s2 in spins:
                for dag1 in [False, True]:
                    for dag2 in [False, True]:
                        op1 = c_dag(j1, s1, n, J) if dag1 else c(j1, s1, n, J)
                        op2 = c_dag(j2, s2, n, J) if dag2 else c(j2, s2, n, J)
                        anticommutator = op1 @ op2 + op2 @ op1
                        cmp = jnp.eye(op1.shape[0], dtype=complex)*int((j1 == j2) and (s1 == s2) and (dag1 != dag2))  # {c, c†} = 1 iff same site and spin and different daggers, else 0
                        assert jnp.allclose(anticommutator, cmp), f"Anticommutation relation failed for sites {j1}, {j2}, spins {s1}, {s2}."
print("Test 1 passed: Anticommutation relations hold.")

# Test 2: Check that the two Hamiltonians are identical
print("Running Test 2: Hamiltonian equivalence...")
assert jnp.allclose(H_QS, H_FH), "Hamiltonians are not identical."
print("Test 2 passed: Hamiltonians are identical.")

# Test 3: Check that the initial state has the correct number of particles (0)
print("Running Test 3: Initial state particle number...")
# Test all n_op(j,s) individually and full_n_op
for j in range(n):
    for s in spins:
        num_fermions = psi.T.conj().dot(n_op(j,s,n,J)).dot(psi)
        assert num_fermions == 0, f"Invalid number of fermions {num_fermions} at site {j}, spin {s}."
num_fermions = psi.T.conj().dot(full_n_op(n, J).dot(psi))
assert num_fermions == 0, f"Expected 0 fermions, but found {num_fermions}."
print("Test 3 passed: Initial state has the correct number of particles (0).")

# Test 4: Check creation and annihilation operators on the initial state
print("Running Test 4: Creation and annihilation operators...")
# Create one fermion at each site and spin, then annihilate it again
for j in range(n):
    for s in spins:
        psi_created = c_dag(j, s, n, J).dot(psi)
        num_fermions = psi_created.T.conj().dot(full_n_op(n, J).dot(psi_created))
        assert num_fermions == 1, f"Expected 1 fermion after creation at site {j}, spin {s}, but found {num_fermions}."
        num_at_site = psi_created.T.conj().dot(n_op(j, s, n, J).dot(psi_created))
        assert num_at_site == 1, f"Expected 1 fermion at site {j}, spin {s}, but found {num_at_site}."
        psi_annihilated = c(j, s, n, J).dot(psi_created)
        num_fermions = psi_annihilated.T.conj().dot(full_n_op(n, J).dot(psi_annihilated))
        assert num_fermions == 0, f"Expected 0 fermions after annihilation at site {j}, spin {s}, but found {num_fermions}."
        assert jnp.allclose(psi, psi_annihilated), f"State after creation and annihilation does not match initial state at site {j}, spin {s}."
print("Test 4 passed: Creation and annihilation operators work correctly.")