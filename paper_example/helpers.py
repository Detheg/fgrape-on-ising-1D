# ruff: noqa
import sys, os
from types import NoneType
import numpy as np

sys.path.append(os.path.abspath("./../feedback-grape"))
sys.path.append(os.path.abspath("./../"))

# ruff: noqa
from feedback_grape.fgrape import Gate, Decay # type: ignore
from feedback_grape.utils.states import basis # type: ignore
from feedback_grape.utils.fidelity import ket2dm, fidelity # type: ignore
from feedback_grape.utils.operators import sigmaz, sigmam # type: ignore
from feedback_grape.utils.modeling import embed # type: ignore
import dynamiqs as dq
import jax.numpy as jnp
import jax
import numpy as np
jax.config.update("jax_enable_x64", True)

# All the operators we need
def generate_hermitian(params, dim):
    assert len(params) == dim**2, "Number of real parameters must be dim^2 for an NxN Hermitian matrix."
    
    X = params.reshape(dim, dim)

    # Real part: take the symmetric part of X
    Re = 0.5 * (X + X.T)

    # Imag part: take the antisymmetric part of X
    Im = 0.5 * (X - X.T)

    # Build Hermitian matrix: Herm = Re + 1j * Im
    H = Re + 1j * Im

    return H

generate_hermitian = jax.jit(generate_hermitian, static_argnames=['dim'])

def generate_unitary(params, dim):
    assert len(params) == dim**2, "Number of real parameters must be dim^2 for an NxN unitary matrix."

    H = generate_hermitian(params, dim)
    return jax.scipy.linalg.expm(-1j * H)
generate_unitary = jax.jit(generate_unitary, static_argnames=['dim'])

def generate_povm(measurement_outcome, params, dim):
    """ 
        Generate a 2-outcome POVM elements M_0 and M_1 for a system with Hilbert space dimension dim.
        This function should parametrize all such POVMs up to unitary equivalence, i.e., M_i -> U M_i for some unitary U.
        I.e it parametrizes all pairs (M_0, M_1) such that M_0 M_0† + M_1 M_1† = I.

        measurement_outcome: 0 or 1, indicating which POVM element to generate.
        params: list of dim^2 real parameters.

        when measurement_outcome == 1:
            M_1 = S D S†
        when measurement_outcome == -1:
            M_0 = S (I - D) S†

        where S is a unitary parametrized by dim^2 parameters, and D is a diagonal matrix with eigenvalues parametrized by dim parameters.
    """
    assert len(params) == dim * (dim + 1), "Number of real parameters must be N * (N + 1) for an NxN POVM element."

    S = generate_unitary(params[0:dim*dim], dim=dim) # All parameters for unitary

    d_vec = jnp.astype(jnp.square(jnp.sin( params[dim*dim:dim*(dim+1)] )), jnp.complex128) # Last #dim parameters for eigenvalues
    d_vec = 1e-8 + (1 - 2e-8) * d_vec # Avoid exactly 0 or 1 eigenvalues

    # jnp.multiply is fast way to "matrix @ diagonal matrix" multiplication, especially for large matrices
    return jnp.where(measurement_outcome == 1,
        jnp.multiply(S, d_vec) @ S.conj().T,
        jnp.multiply(S, jnp.sqrt(1 - jnp.square(d_vec))) @ S.conj().T
    )
generate_povm = jax.jit(generate_povm, static_argnames=['dim'])

def __jump_ops(N_qubits, gamma_z, gamma_m):
    return [
        gamma_z**0.5 * embed(sigmaz(), 1, (2**j, 2, 2**(N_qubits - j - 1)))
        for j in range(N_qubits)
    ] + [
        gamma_m**0.5 * embed(sigmam(), 1, (2**j, 2, 2**(N_qubits - j - 1)))
        for j in range(N_qubits)
    ]

def generate_jump_superoperator(N_qubits, gamma_z, gamma_m):
    # 1. Define jump operators for each qubit
    jump_ops = __jump_ops(N_qubits, gamma_z, gamma_m)

    # 2. Define superoperator for lindblad evolution
    N = 2**N_qubits
    O = jnp.zeros((N,N), dtype=jnp.complex128)

    super_op = dq.mepropagator(
        O,
        jump_ops=jump_ops,
        tsave=[1],
        options=dq.Options(t0 = 0)
    ).propagators[0].to_numpy()
    
    for i in range(N*N):
        for j in range(N*N):
            assert np.abs(np.imag(super_op[i,j])) <= 1e-14, f"Superoperator element {super_op[i,j]} is not real."
    
    super_op = jnp.real(super_op)
    super_op = jnp.array(super_op, dtype=jnp.float64)

    def lindblad_solution(rho): # 3. Define function for state transition under lindblad evolution
        return (super_op @ rho.flatten()).reshape(N, N)
    
    return jax.jit(lindblad_solution)

# Functions which initialize gates
def init_decay_gate(N_qubits, gamma_z, gamma_m):
    decay_fun = generate_jump_superoperator(N_qubits, gamma_z, gamma_m)

    decay_gate = Gate(
        gate=lambda rho, _: decay_fun(rho),
        initial_params = jnp.array([]), # No parameters
        measurement_flag = False,
        quantum_channel_flag = True,
    )

    return decay_gate

def init_numeric_decay_gate(N_qubits, gamma_z, gamma_m):
    jump_ops = __jump_ops(N_qubits, gamma_z, gamma_m)

    decay_gate = Decay(
        c_ops=jump_ops,
    )

    return decay_gate

def init_povm_gate(key, N_qubits):
    base_dim = 2**N_qubits
    N_povm_params = base_dim*(base_dim+1)
    povm_gate = Gate(
        gate=lambda msmt, params: generate_povm(msmt, params, dim=base_dim),
        initial_params = jax.random.uniform(key, (N_povm_params,), minval=0.0, maxval=2*jnp.pi),
        measurement_flag = True
    )

    return povm_gate

def init_unitary_gate(key, N_qubits):
    base_dim = 2**N_qubits
    N_unitary_params = base_dim**2
    U_gate = Gate(
        gate=lambda params: generate_unitary(params, dim=base_dim),
        initial_params = jax.random.uniform(key, (N_unitary_params,), minval=0.0, maxval=1.0),
        measurement_flag = False
    )

    return U_gate

def init_identity_gate():
    U_gate = Gate(
        gate=lambda rho, _: rho,
        initial_params = jnp.array([]),
        measurement_flag = False,
        quantum_channel_flag = True,
    )

    return U_gate

# Functions which initialize gate combinations for the protocols
def init_fgrape_protocol(key, N_qubits, N_meas, gamma_z, gamma_m, use_numeric_decay=False):
    subkey1, subkey2 = jax.random.split(key, 2)

    decay_gate = (
        init_numeric_decay_gate(N_qubits, gamma_z, gamma_m)
        if use_numeric_decay else
        init_decay_gate(N_qubits, gamma_z, gamma_m)
    )
    povm_gate = init_povm_gate(subkey1, N_qubits)
    U_gate = init_unitary_gate(subkey2, N_qubits)

    return [decay_gate] + [povm_gate] * N_meas + [U_gate]

def init_fgrape_protocol_wo_decay(key, N_qubits, N_meas, gamma_z, gamma_m):
    subkey1, subkey2 = jax.random.split(key, 2)

    identity_gate = init_identity_gate()
    povm_gate = init_povm_gate(subkey1, N_qubits)
    U_gate = init_unitary_gate(subkey2, N_qubits)

    return [identity_gate] + [povm_gate] * N_meas + [U_gate]

def generate_superposition_state(N_qubits):
    psi_zero = basis(2**N_qubits, 0)
    psi_one = basis(2**N_qubits, 2**N_qubits - 1)

    psi = (psi_zero + psi_one) / jnp.sqrt(2)
    rho = ket2dm(psi)

    return rho


# Tests for the implementations
def test_implementations():
    # Test unitary and special unitary generators
    for i in range(10):
        key = jax.random.PRNGKey(i)
        key, subkey1 = jax.random.split(key, 2)

        params = jax.random.uniform(subkey1, (16,), minval=0.0, maxval=2*jnp.pi)

        U = generate_unitary(params, 4)

        assert jnp.allclose(U @ U.conj().T, jnp.eye(4)), "Unitary condition failed"

    # Test povm generator
    for i in range(10):
        key = jax.random.PRNGKey(i)
        key, subkey = jax.random.split(key, 2)

        for f,N_params in [(lambda msmt, params: generate_povm(msmt, params, 4), 4*(4+1))]:
            params = jax.random.uniform(subkey, (N_params,), minval=0.0, maxval=2*jnp.pi)

            M_0 = f(-1, params)
            M_1 = f(1, params)

            assert jnp.allclose(M_0 @ M_0.conj().T + M_1 @ M_1.conj().T, jnp.eye(4)), "POVM elements do not sum to identity."

# Baseline calculation
def calculate_baseline(N_qubits: int, gamma_z: float, gamma_m: float, evaluation_time_steps: int):
    decay_superoperator = generate_jump_superoperator(N_qubits, gamma_z, gamma_m)

    # Evolve basis state and compute fidelities
    state = generate_superposition_state(N_qubits)
    target_state = state.copy()

    fidelities_each = np.zeros(evaluation_time_steps+1)
    fidelities_each[0] = fidelity(
        C_target=target_state,
        U_final=state,
        evo_type="density",
    )

    def propagate_single_timestep(rho, rho_target):
        tmp = decay_superoperator(rho)

        fid = fidelity(C_target=rho_target, U_final=tmp, evo_type="density")

        return tmp, fid

    f = jax.jit(lambda rho: propagate_single_timestep(rho, target_state))

    states_each = [state.copy()]
    for i in range(evaluation_time_steps):
        state, fid = f(state)
        fidelities_each[i+1] = fid

        assert np.all(np.isclose(state, state.conj().T)), "State is not Hermitian"
        assert np.isclose(np.trace(state).real, 1.0), "State is not normalized"
        assert np.all(np.linalg.eigvalsh(state) >= -1e-10), "State is not positive semidefinite"
        states_each.append(state.copy())

    return fidelities_each, states_each

# Convert JAX/NumPy arrays to lists for JSON serialization
def __to_serializable(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: __to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [__to_serializable(x) for x in obj]
    return obj

# Convert FgResult to dictionary for JSON serialization
def FgResult_to_dict(result):
    data = {
        "iterations": result.iterations,
        #"returned_params": __to_serializable(result.returned_params), # Removed because it takes too much space
        "final_fidelity": __to_serializable(result.final_fidelity) if type(result.final_fidelity) != NoneType else -1,
        "final_purity":  __to_serializable(result.final_purity) if type(result.final_purity) != NoneType else -1,
        "optimized_trainable_parameters": __to_serializable(result.optimized_trainable_parameters),
        "fidelity_each_timestep": __to_serializable(np.array(result.fidelity_each_timestep)),
        #"state_each_timestep_real": __to_serializable(np.real(np.array(result.state_each_timestep))), # Removed because it takes too much space
        #"state_each_timestep_imag": __to_serializable(np.imag(np.array(result.state_each_timestep))),
    }

    return data

experiment_param_formats = [
    ("t", int),
    ("w", list),
    ("Nqubits", int),
    ("Nmeas", int),
    ("gammaz", float),
    ("gammam", float),
    ("Ntrainiter", int),
    ("rnnhsize", int),
    ("learning_rate", float),
]