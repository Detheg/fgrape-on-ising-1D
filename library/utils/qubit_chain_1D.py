# Add the feedback-grape git submodule to the path
import sys, os
sys.path.append(os.path.abspath("./feedback-grape"))

from feedback_grape.utils.operators import (
    sigmax,
    sigmay,
    sigmaz,
    identity,
)
from feedback_grape.utils.states import basis
from feedback_grape.utils.tensor import tensor
from feedback_grape.utils.fidelity import fidelity
from jax import numpy as jnp
from jax.scipy.linalg import expm

def embed(op: jnp.ndarray, j: int, n: int) -> jnp.ndarray:
    """
    Embed a single-qubit operator `op` into an `n`-qubit Hilbert space at position `j`.

    Parameters:
    op (np.ndarray): The single-qubit operator to embed.
    j (int): The position (0-indexed) to embed the operator.
    n (int): The total number of qubits.

    Returns:
    np.ndarray: The embedded operator in the n-qubit Hilbert space.
    """
    if j < 0 or j >= n:
        raise ValueError("Index j must be in the range [0, n-1].")
    if op.shape != (2, 2):
        raise ValueError("Operator op must be a 2x2 matrix.")
    

    ops = [identity(2)] * n
    ops[j] = op
    return tensor(*ops)

def ising_model_1D_Hamiltonian(n: int, J: jnp.ndarray, h: float, boundary="open") -> jnp.ndarray:
    """
    Construct the 1D Ising model Hamiltonian for n qubits with coupling J and transverse field h.

    Parameters:
    n (int): Number of qubits.
    J (jnp.ndarray): Coupling strength between neighboring qubits.
    h (float): Transverse magnetic field strength.
    boundary (str): 'open' for open boundary conditions, 'periodic' for periodic boundary conditions.

    Returns:
    np.ndarray: The Hamiltonian matrix of the 1D Ising model.
    """
    if boundary not in ["open", "periodic"]:
        raise ValueError("Invalid boundary condition. Choose 'open' or 'periodic'.")
    if len(J) != n - 1 and boundary == "open":
        raise ValueError("Length of J must be n-1 for open boundary conditions.")
    if len(J) != n and boundary == "periodic":
        raise ValueError("Length of J must be n for periodic boundary conditions.")
    if n < 1:
        raise ValueError("Number of qubits n must be at least 1.")

    H = jnp.zeros((2**n, 2**n), dtype=complex)

    # Interaction term
    for i in range(n - int(boundary == "open")):
        H += -0.5 * J[i] * embed(sigmaz(), i, n) @ embed(sigmaz(), (i + 1)%n, n)

    # Transverse field term
    for i in range(n):
        H += -h * embed(sigmaz(), i, n)

    return H


# Control operators (X,Y interaction between neighboring qubits and X rotation on last qubit which is the minimum control set to achieve full controllability)
# Measurement operator (projective measurement along z-axis on first qubit)
def projection_measurement_operator(measurement_outcome, n):
    return jnp.where(
        measurement_outcome == 1,
        tensor(jnp.eye(2**(n-1)), basis(2, 0)@basis(2, 0).conj().T),
        tensor(jnp.eye(2**(n-1)), basis(2, 1)@basis(2, 1).conj().T),
    )
    
def unitary_op(params, n):
    transport_params = params[1:]
    rotation_params  = params[0]

    def transport_unitary(params):
        J_x, J_y = params
        return expm(-1j*sum([
            -0.5*J_x*embed(sigmax(), i, n)@embed(sigmax(), i+1, n)
            -0.5*J_y*embed(sigmay(), i, n)@embed(sigmay(), i+1, n)
            for i in range(n-1)
        ]))

    def rotation_unitary(params):
        alpha = params
        return expm(-1j*embed(sigmax(), n-1, n)*alpha) # Local X rotation on last qubit

    return transport_unitary(transport_params) @ rotation_unitary(rotation_params)

def calculate_expected_fidelity(lookup_table, t_trained, n, t, rho_initial, rho_target, t_now=0, history=0, fidelity_history=None, P_branch=1.0):
    if fidelity_history is None:
        fidelity_history = [0]*t
    if t_now < t_trained:
        row = t_now
    else:
        row = t_trained - 1

    history = (history << 1) & ((1 << t_trained) - 1) # Keep only last t_trained bits

    fidelity_ = 0.0
    P_tmp = 0.0
    for meas in [1, -1]:
        op = unitary_op(lookup_table[row][history | (1 if meas == -1 else 0)], n)
        M  = projection_measurement_operator(meas, n)
        rho_new = M @ rho_initial @ M.conj().T
        P = jnp.real(jnp.trace(rho_new))
        if P > 1e-12:
            rho_new /= P
            rho_new = op @ rho_new @ op.conj().T
            
            P_tmp += P

            assert jnp.isclose(jnp.trace(rho_new), 1.0), f"Trace is not 1 but {jnp.trace(rho_new)}"

            fidelity_history[t_now] += P*fidelity(C_target=rho_target, U_final=rho_new, evo_type="density")

            if t_now < t - 1:
                calculate_expected_fidelity(
                    lookup_table,
                    t_trained,
                    n,
                    t,
                    rho_new,
                    rho_target,
                    t_now + 1,
                    (history | (1 if meas == -1 else 0)),
                    fidelity_history,
                    P_branch*P
                )
    
    assert jnp.isclose(P_tmp, 1.0), f"Total probability is not 1 but {P_tmp}"

    if t_now == 0:
        # Normalize by number of branches
        for i in range(t):
            fidelity_history[i] /= 2**i
        return fidelity_history
    else:
        return []