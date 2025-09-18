# Add the feedback-grape git submodule to the path
import sys, os
sys.path.append(os.path.abspath("./feedback-grape"))
print(os.path.abspath("./feedback-grape"))

from feedback_grape.utils.tensor import tensor
from feedback_grape.utils.states import basis
from feedback_grape.utils.operators import identity, sigmam, sigmap
from jax import numpy as jnp
from scipy.linalg import expm

def FH_zero_particle_state(N_sites: int, J: float):
    """
    Initialize the Heisenberg model ket state |psi> to emulate a spin-J 1D
    fermionic chain with N_sites fermionic sites without any filled states.

    In case spin J=0.0 the system is emulated on a 1D single qubit chain.
    If J=0.5 it is mapped on a double chain whereas the second chain is tensored
    at the end of the single chain system. The first half then corresponds to spin-down, the second to spin-up.

    Parameters:
    N_sites (int): Number of sites in the chain.
    J (float): Spin.

    Returns:
    np.ndarray: The ket state of the Heisenberg model.
    """

    # Check input validity
    _check_jsNJ(0, J, N_sites, J) # Just to validate N_sites and J
    
    # Initialize wavefunction of ground state (#fermions = 0)
    ground_state = basis(2, 1) # |psi_i> = (0, 1) corresponds to single lattice point without fermion
    N_qubits = int(N_sites * (2*J + 1))

    return tensor(*([ground_state]*N_qubits))

def c(j, s, N_sites, J):
    """
    Annihilation operator c_j,s for fermion at site j with spin s.

    Parameters:
    j (int): Site index (0 <= j < N_sites).
    s (float): Spin index (-J <= s <= J).
    N_sites (int): Number of sites in the chain.
    J (float): Spin.

    Returns:
    np.ndarray: The creation operator matrix.
    """

    # Check input validity
    _check_jsNJ(j, s, N_sites, J)
    
    N_qubits = int(N_sites * (2*J + 1))
    
    # Perform Jordan-Wigner transformation
    k = _js_to_k(j,s,N_sites,J)
    lam = jnp.zeros((2**N_qubits,2**N_qubits), dtype=complex)

    for l in range(0,k):
        sig1 = _map_spinop_to_HS(sigmap(), l, N_qubits)
        sig2 = _map_spinop_to_HS(sigmam(), l, N_qubits)

        lam += sig1 @ sig2

    phase = 1j*jnp.pi*lam
    c_js = expm(phase) @ _map_spinop_to_HS(sigmam(), k, N_qubits)

    return c_js

def c_dag(j, s, N_sites, J):
    """
    Creation operator c†_j,s for fermion at site j with spin s.

    Parameters:
    j (int): Site index (0 <= j < N_sites).
    s (float): Spin index (-J <= s <= J).
    N_sites (int): Number of sites in the chain.
    J (float): Spin.

    Returns:
    np.ndarray: The creation operator matrix.
    """

    return c(j, s, N_sites, J).T.conj()

def n_op(j, s, N_sites, J):
    """
    Number operator n_j,s = c†_j,s c_j,s for fermion at site j with spin s.

    Parameters:
    j (int): Site index (0 <= j < N_sites).
    s (float): Spin index (-J <= s <= J).
    N_sites (int): Number of sites in the chain.
    J (float): Spin.

    Returns:
    np.ndarray: The number operator matrix.
    """

    return c_dag(j, s, N_sites, J) @ c(j, s, N_sites, J)

def full_n_op(N_sites, J):
    """
    Full number operator N = sum_{j,s} n_j,s for fermions in the chain.

    Parameters:
    N_sites (int): Number of sites in the chain.
    J (float): Spin.

    Returns:
    np.ndarray: The full number operator matrix.
    """

    # Check input validity
    _check_jsNJ(0, J, N_sites, J)  # Just to validate N_sites and J

    N_qubits = int(N_sites * (2*J + 1))
    full_n_op = jnp.zeros((2**N_qubits, 2**N_qubits), dtype=complex)

    if J == 0.0:   spins = [0]
    elif J == 0.5: spins = [-0.5, 0.5]

    for j in range(0, N_sites):
        for s in spins:
            full_n_op += n_op(j, s, N_sites, J)

    return full_n_op

def embed(operator, j, s, N_sites, J):
    """
      Maps a 2x2 matrix acting on single qubit representing fermion at index j with spin s to Hilbert space of Heisenberg model.
    """
    # Check input validity
    _check_jsNJ(j, s, N_sites, J)

    # Index of qubit in Heisenberg model
    N_qubits = int(N_sites * (2*J + 1))
    k = _js_to_k(j,s,N_sites,J)
    
    return _map_spinop_to_HS(operator, k, N_qubits)

def emulated_FH_Hamiltonian(N_sites, J, mu, t, U, boundary='open'):
    """
    Constructs the Fermi-Hubbard Hamiltonian for a 1D chain of fermions with N_sites and spin J.

    Parameters:
    N_sites (int): Number of sites in the chain.
    J (float): Spin.
    mu (float): Chemical potential.
    t (float): Transfer parameter.
    U (float): On-site interaction strength.
    boundary (str): 'open' for open boundary conditions, 'periodic' for periodic boundary conditions.

    Returns:
    np.ndarray: The Fermi-Hubbard Hamiltonian matrix.
    """

    # Check input validity
    _check_jsNJ(0, J, N_sites, J)  # Just to validate N_sites and J
    if type(mu) not in [int, float]:
        raise TypeError("mu must be a number.")
    if type(t) not in [int, float]:
        raise TypeError("t must be a number.")
    if type(U) not in [int, float]:
        raise TypeError("U must be a number.")
    if boundary not in ['open', 'periodic']:
        raise ValueError("Invalid boundary condition.")

    N_qubits = int(N_sites * (2*J + 1))
    H_FH = (
        -mu*full_n_op(N_sites, J)
        -t*sum(c_dag(j, s, N_sites, J)@c((j+1)%N_sites, s, N_sites, J) + c_dag((j+1)%N_sites, s, N_sites, J)@c(j, s, N_sites, J) for j in range(N_sites-(boundary=='open')) for s in ([-0.5, 0.5] if J==0.5 else [0]))
    )

    if J != 0.0: # Onsite interaction only for spinless system
        H_FH += +U*sum(c_dag(j, 0.5, N_sites, J)@c(j, 0.5, N_sites, J)@c_dag(j, -0.5, N_sites, J)@c(j, -0.5, N_sites, J) for j in range(N_sites))

    return H_FH

def FH_energy_shift(N_sites, J, mu, U):
    """
    Calculate the constant energy shift for the Fermi-Hubbard Hamiltonian.

    Parameters:
    N_sites (int): Number of sites in the chain.
    J (float): Spin.
    mu (float): Chemical potential.
    U (float): On-site interaction strength.

    Returns:
    np.ndarray: The constant energy shift matrix.
    """

    # Check input validity
    _check_jsNJ(0, J, N_sites, J)  # Just to validate N_sites and J
    if type(mu) not in [int, float]:
        raise TypeError("mu must be a number.")
    if type(U) not in [int, float]:
        raise TypeError("U must be a number.")

    N_qubits = int(N_sites * (2*J + 1))
    Delta_E = identity(2**N_qubits) * (-0.5*N_sites*(2*J+1)*mu + 0.25*N_sites*U*int(J != 0))  # Constant energy shift not mentioned in literature

    return Delta_E

def _map_spinop_to_HS(operator, k, N_qubits):
    """
      Maps a 2x2 matrix acting on single qubit at index k to Hilbert space of N_qubit system.
    """
    if k == 0:
        return tensor(operator, identity(2**(N_qubits-k-1)))
    elif k == N_qubits - 1:
        return tensor(identity(2**k), operator)
    else:
        return tensor(identity(2**k), operator, identity(2**(N_qubits-k-1)))

def density_matrix_size_bytes(N_sites, J):
    """
    Calculate the size of the density matrix in bytes for a fermionic chain with N_sites and spin J.

    Parameters:
    N_sites (int): Number of sites in the chain.
    J (float): Spin.

    Returns:
    float: Size of the density matrix in bytes.
    """

    # Check input validity
    _check_jsNJ(0, J, N_sites, J)  # Just to validate N_sites and J

    N_qubits = int(N_sites * (2*J + 1))
    size_bytes = (2**N_qubits)**2 * 16  # Each complex number takes 16 bytes (8 bytes for real and 8 for imag)

    return size_bytes

def _js_to_k(j,s,N_sites,J):
    _check_jsNJ(j, s, N_sites, J)

    # Convert index j,s in fermionic chain to 1-d numbering on spin chain
    if J == 0 and s == 0:
        return j
    elif J == 0.5 and s in [-0.5, 0.5]:
        if s == -0.5:
            return j
        elif s == 0.5:
            return j + N_sites
    else:
        raise ValueError("Invalid spin value for given J.")
    return 0 # Should never reach here

def _check_jsNJ(j, s, N_sites, J):
    # Check input validity
    if type(j) is not int:
        raise TypeError("j must be an integer.")
    if type(N_sites) is not int:
        raise TypeError("N_sites must be an integer.")
    if j < 0 or j >= N_sites:
        raise ValueError("j must be in the range [0, N_sites-1].")
    if type(s) not in [int, float]:
        raise TypeError("s must be a number.")
    if J == 0 and s != 0:
        raise ValueError("For J=0, s must be 0.")
    if J == 0.5 and s not in [-0.5, 0.5]:
        raise ValueError("For J=1/2, s must be -1/2 or 1/2.")
    if N_sites < 1:
        raise ValueError("N_sites must be at least 1.")
    if type(J) not in [int, float]:
        raise TypeError("J must be a number.")
    if J not in [0.0, 0.5]: 
        raise ValueError("J must be either 0 or 1/2.")
    
