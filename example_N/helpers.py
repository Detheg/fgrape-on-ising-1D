# ruff: noqa
import sys, os
sys.path.append(os.path.abspath("./../feedback-grape"))
sys.path.append(os.path.abspath("./../"))

# ruff: noqa
from feedback_grape.fgrape import Gate # type: ignore
from feedback_grape.utils.states import basis # type: ignore
from feedback_grape.utils.fidelity import ket2dm, fidelity # type: ignore
from feedback_grape.utils.operators import sigmap, sigmam # type: ignore
from typing import Callable
import dynamiqs as dq
import jax.numpy as jnp
import jax
from tqdm import tqdm
import numpy as np
from library.utils.qubit_chain_1D import embed

jax.config.update("jax_enable_x64", True)

# All the operators we need
def generate_traceless_hermitian(params, dim):
    assert len(params) == dim**2 - 1, "Number of real parameters must be dim^2 - 1 for an NxN traceless Hermitian matrix."
    
    # Read the first (dim**2 - dim) / 2 as the real parts of the upper triangle
    real_parts = jnp.array(params[: (dim**2 - dim) // 2])

    # Read the next (dim**2 - dim) / 2 as the imaginary parts of the upper triangle
    imag_parts = jnp.array(params[(dim**2 - dim) // 2 : - (dim - 1)])

    # Read the last (dim - 1) as the diagonal elements, set the last diagonal element to ensure tracelessness
    trace = sum(params[- (dim - 1):])
    diag_parts = jnp.append(params[- (dim - 1):], jnp.array([-trace]))

    # Construct the Hermitian matrix
    triag_parts = real_parts + 1j * imag_parts

    return jnp.array([
        [
            diag_parts[i] if i == j else
            triag_parts[(i * (i - 1)) // 2 + j - i - 1] if i < j else
            jnp.conj(triag_parts[(j * (j - 1)) // 2 + i - j - 1])
            for j in range(dim)
        ] for i in range(dim)
    ])
generate_traceless_hermitian = jax.jit(generate_traceless_hermitian, static_argnames=['dim'])

def generate_hermitian(params, dim):
    assert len(params) == dim**2, "Number of real parameters must be dim^2 for an NxN Hermitian matrix."
    
    # Generate traceless hermitanian from first dim^2 - 1 parameters and read last parameter as trace
    return generate_traceless_hermitian(params[:-1], dim) + jnp.eye(dim) * params[-1] / dim
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

    d_vec = jnp.astype(jnp.sin( params[dim*dim:dim*(dim+1)] ) ** 2, jnp.complex128) # Last #dim parameters for eigenvalues
    d_vec = 1e-8 + (1 - 2e-8) * d_vec # Avoid exactly 0 or 1 eigenvalues

    return jnp.where(measurement_outcome == 1,
        S @ jnp.diag(d_vec) @ S.conj().T,
        S @ jnp.diag(jnp.sqrt(1 - d_vec**2)) @ S.conj().T
    )
generate_povm = jax.jit(generate_povm, static_argnames=['dim'])

def generate_jump_superoperator(N_qubits, gamma_p, gamma_m):
    # 1. Define jump operators for each qubit
    jump_ops = [
        gamma_p**0.5 * embed(sigmap(), j, N_qubits)
        for j in range(N_qubits)
    ] + [
        gamma_m**0.5 * embed(sigmam(), j, N_qubits)
        for j in range(N_qubits)
    ]

    # 2. Define superoperator for lindblad evolution
    N = 2**N_qubits
    I = jnp.eye(N, dtype=jnp.complex128)
    O = jnp.zeros((N,N), dtype=jnp.complex128)

    super_op = dq.mepropagator(O, jump_ops=jump_ops, tsave=[0, 1]).propagators[1].to_jax()

    # 3. Define function for state transition under lindblad evolution
    def lindblad_solution(rho):
        return (super_op @ rho.flatten()).reshape(N, N)

    return jax.jit(lindblad_solution)

def generate_jump_superoperator_single_qubit(gamma_p, gamma_m):
    gamma = gamma_p + gamma_m
    p = gamma_p / gamma if gamma > 0 else 0.5
    q = np.exp(-gamma)

    # 3. Define function for state transition under lindblad evolution
    def lindblad_solution(rho):
        return jnp.array([
            [p + q*(rho.at[0,0].get() - p), q**0.5 * rho.at[0,1].get()],
            [q**0.5 * rho.at[1,0].get(), (1 - p) - q*(rho.at[0,0].get() - p)],
        ])
    return jax.jit(lindblad_solution)

# Functions which initialize gates
def init_decay_gate(N_qubits, gamma_p, gamma_m):
    decay_fun = generate_jump_superoperator(N_qubits, gamma_p, gamma_m)

    decay_gate = Gate(
        gate=lambda rho, _: decay_fun(rho),
        initial_params = jnp.array([]), # No parameters
        measurement_flag = False,
        quantum_channel_flag = True,
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

# Functions which initialize gate combinations for the protocols
def init_fgrape_protocol(key, N_qubits, N_meas, gamma_p, gamma_m):
    subkey1, subkey2 = jax.random.split(key, 2)

    decay_gate = init_decay_gate(N_qubits, gamma_p, gamma_m)
    povm_gate = init_povm_gate(subkey1, N_qubits)
    U_gate = init_unitary_gate(subkey2, N_qubits)

    return [decay_gate] + [povm_gate] * N_meas + [U_gate]

# Function to generate random states
def generate_random_bloch_state(key, N_qubits):
    """
        Generate a random density matrix on Bloch sphere (cos(theta/2)*|00 ... 0> + sin(theta/2)*exp(i*phi)*sin(theta/2)*|11 ... 1>)
        and theta, phi are drawn from uniform distributions.
    """
    
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    theta = jax.random.uniform(subkey1, minval=0.0, maxval=jnp.pi)
    phi = jax.random.uniform(subkey2, minval=0.0, maxval=2*jnp.pi)
    alpha = jnp.cos(theta / 2)
    beta = jnp.sin(theta / 2) * jnp.exp(1j * phi)

    psi_one  = basis(2**N_qubits, 0)
    psi_zero = basis(2**N_qubits, 2**N_qubits - 1)

    psi = (
        alpha * psi_zero
        + beta * psi_one
    )

    return ket2dm(psi)

def generate_random_discrete_state(key, N_qubits):
    """
        Generate a random density matrix which is either |00 .. 0><00 .. 0| or |11 .. 1><11 .. 1|.
    """

    subkey1, _ = jax.random.split(key, 2) # 2 keys for consistency with generate_random_bloch_state

    psi_one  = basis(2**N_qubits, 0)
    psi_zero = basis(2**N_qubits, 2**N_qubits - 1)

    psi = jnp.where(jax.random.uniform(subkey1) < 0.5,
        psi_zero,
        psi_one
    )

    return ket2dm(psi)

def generate_excited_state(key, N_qubits):
    """
        Generate the excited state |11 ... 1><11 ... 1|.
    """

    psi_one  = basis(2**N_qubits, 0)

    return ket2dm(psi_one)

def generate_ground_state(key, N_qubits):
    """
        Generate the ground state |00 ... 0><00 ... 0|.
    """

    psi_zero = basis(2**N_qubits, 2**N_qubits - 1)

    return ket2dm(psi_zero)

# Tests for the implementations
def test_implementations():
    # Test random state generation
    for callab in [generate_random_discrete_state, generate_random_bloch_state, generate_excited_state, generate_ground_state]:
        for i in range(10):
            key = jax.random.PRNGKey(i)
            rho = callab(key, N_qubits=2)

            assert jnp.allclose(rho, rho.conj().T), "Generated state is not Hermitian"
            assert jnp.isclose(jnp.trace(rho), 1.0), "Generated state does not have trace 1"
            eigvals = jnp.linalg.eigvals(rho)
            assert jnp.all(eigvals >= -1e-10), "Generated state is not positive semidefinite"


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

            assert jnp.allclose(M_0 @ M_0.conj().T + M_1 @ M_1.conj().T, jnp.eye(4)), "POVM elements do not sum to identity"
            assert jnp.allclose(M_0, M_0.conj().T), "POVM element M_0 is not Hermitian"
            assert jnp.allclose(M_1, M_1.conj().T), "POVM element M_1 is not Hermitian"
            assert jnp.all(jnp.linalg.eigvals(M_0) >= 0), "POVM element M_0 is not positive semidefinite"
            assert jnp.all(jnp.linalg.eigvals(M_1) >= 0), "POVM element M_1 is not positive semidefinite"

def calculate_baseline(N_qubits: int, gamma_p: float, gamma_m: float, evaluation_time_steps: int, batch_size: int, generate_state: callable, key=jax.random.PRNGKey(0)):
    decay_superoperator = generate_jump_superoperator(N_qubits, gamma_p, gamma_m)

    # Evolve all basis states and compute fidelities
    keys = jax.random.split(key, batch_size)
    states = jnp.array([generate_state(key, N_qubits) for key in keys])
    target_states = jnp.array([generate_state(key, N_qubits) for key in keys])
    states_each = [states.copy()]

    fidelities_each = np.zeros((len(states), evaluation_time_steps+1))
    for i, (state, target_state) in enumerate(zip(states, target_states)):
        fidelities_each[i, 0] = fidelity(
            C_target=target_state,
            U_final=state,
            evo_type="density",
        )

    def propagate_single_timestep(rho, rho_target):
        tmp = decay_superoperator(rho)

        fid = fidelity(C_target=rho_target, U_final=tmp, evo_type="density")

        return tmp, fid

    propagate_single_timestep_vmap = jax.vmap(jax.jit(propagate_single_timestep))

    for i in range(evaluation_time_steps):
        states, fid = propagate_single_timestep_vmap(states, target_states)
        fidelities_each[:, i+1] = fid

        for j, rho in enumerate(states):
            assert np.all(np.isclose(rho, rho.conj().T)), "State is not Hermitian"
            assert np.isclose(np.trace(rho).real, 1.0), "State is not normalized"
            assert np.all(np.linalg.eigvalsh(rho) >= -1e-10), "State is not positive semidefinite"

        states_each.append(states.copy())

    return fidelities_each, states_each

def calculate_baseline_single_qubit(gamma_p: float, gamma_m: float, evaluation_time_steps: int, batch_size: int, generate_state: callable, key=jax.random.PRNGKey(0)):
    decay_superoperator = generate_jump_superoperator_single_qubit(gamma_p, gamma_m)

    # Evolve all basis states and compute fidelities
    keys = jax.random.split(key, batch_size)
    states = jnp.array([generate_state(key, N_qubits=1) for key in keys])
    target_states = jnp.array([generate_state(key, N_qubits=1) for key in keys])
    states_each = [states.copy()]

    fidelities_each = np.zeros((len(states), evaluation_time_steps+1))
    for i, (state, target_state) in enumerate(zip(states, target_states)):
        fidelities_each[i, 0] = fidelity(
            C_target=target_state,
            U_final=state,
            evo_type="density",
        )

    def propagate_single_timestep(rho, rho_target):
        tmp = decay_superoperator(rho)

        fid = fidelity(C_target=rho_target, U_final=tmp, evo_type="density")

        return tmp, fid

    propagate_single_timestep_vmap = jax.vmap(jax.jit(propagate_single_timestep))

    for i in range(evaluation_time_steps):
        states, fid = propagate_single_timestep_vmap(states, target_states)
        fidelities_each[:, i+1] = fid

        for j, rho in enumerate(states):
            assert np.all(np.isclose(rho, rho.conj().T)), "State is not Hermitian"
            assert np.isclose(np.trace(rho).real, 1.0), "State is not normalized"
            assert np.all(np.linalg.eigvalsh(rho) >= -1e-10), "State is not positive semidefinite"

        states_each.append(states.copy())

    return fidelities_each, states_each

# Functions to test a custom protocol definition
# Validate parameters
def __validate_protocol(protocol, N_qubits):
    N = 2**N_qubits

    def check_povm(povm_p, povm_m, N):
        assert povm_p.shape == (N,N), "Povm operator has incorrect shape."
        assert povm_m.shape == (N,N), "Povm operator has incorrect shape."
        assert jnp.allclose(povm_p.conj().T @ povm_p + povm_m.conj().T @ povm_m, jnp.eye(N)), "Povm operators do not sum to identity."
    def check_unitary(U, N):
        assert U.shape == (N,N), "Unitary operator has incorrect shape."
        assert jnp.allclose(U.conj().T @ U, jnp.eye(N)), "Unitary operator is not unitary."

    for key, value in protocol.items():
        if key[:4] == "povm" and key[-1] == "+":
            check_povm(value, protocol[f"{key[:-1]}-"], N)
        elif key[0] == "U":
            check_unitary(value, N)

# Construct lookup table for protocol
def lut_from_protocol(protocol, N_qubits, N_meas):
    __validate_protocol(protocol, N_qubits)

    povm1_init_p = protocol["povm1_init_+"]
    povm1_init_m = protocol["povm1_init_-"]

    N = 2**N_qubits

    initial_params = (
        [[]] # Placeholder for Quantum channel with no parameters
        + [jnp.array(povm1_init_p.real.flatten().tolist() + povm1_init_p.imag.flatten().tolist() + povm1_init_m.real.flatten().tolist() + povm1_init_m.imag.flatten().tolist())] # First POVM (the only one which is applied at the start)
        + [jnp.zeros(N*N*4)]*(N_meas - 1) # placeholder POVMs which are not applied
        + [jnp.zeros(N*N*2)] # placeholder unitary which is not applied at the start
    )

    lookup_table = []
    for i in range(N_meas):
        n = i+1
        col = []
        for meas_hist in range(2**n):
            meas_bin = bin(meas_hist)[2:].zfill(n).replace('1', '-').replace('0', '+')
            params = []
            
            for povm_idx in range(N_meas):
                if n == povm_idx or n == N_meas:
                    povm_p = protocol[f"povm{povm_idx+1}_{meas_bin}+"]
                    povm_m = protocol[f"povm{povm_idx+1}_{meas_bin}-"]
                else:
                    # Not enough measurements yet for this POVM, fill with zeros
                    povm_p = jnp.zeros((N,N))+1
                    povm_m = jnp.zeros((N,N))+1

                params.extend([
                    jnp.array(op.real.flatten().tolist() + op.imag.flatten().tolist())
                    for op in [povm_p, povm_m]
                ])

            if n < N_meas:
                # Not enough measurements yet for this unitary, fill with zeros because it wont be applied
                U = jnp.zeros((N,N))
            else:
                U = protocol[f"U_{meas_bin}"]

            params.append(jnp.array(U.real.flatten().tolist() + U.imag.flatten().tolist()))
            flattened_params = jnp.concatenate(params)
            col.append(flattened_params)

        col.extend([jnp.array([0]*len(flattened_params))]*(2**(N_meas) - 2**(n))) # Fill up to full size with zeros
        lookup_table.append(col)
    
    lut = {
        "initial_params": initial_params,
        "lookup_table": lookup_table,
    }
    return lut

def init_system_params_for_custom_protocol(N_qubits, N_meas, gamma_p, gamma_m):
    N = 2**N_qubits

    @jax.jit
    def unitary_fun(params):
        real = params[:(N*N)].reshape((N,N))
        imag = params[(N*N):].reshape((N,N))
        return real + 1j*imag

    @jax.jit
    def povm_fun(meas, params):
        start = (meas == -1) * 2 * N*N
        real = jax.lax.dynamic_slice(params, (start,), (N*N,)).reshape((N,N))
        imag = jax.lax.dynamic_slice(params, (start + N*N,), (N*N,)).reshape((N, N))
        return real + 1j*imag

    decay_gate = init_decay_gate(N_qubits, gamma_p, gamma_m)

    povm_gate = Gate(
        gate=povm_fun,
        initial_params = jnp.concatenate([jnp.eye(N).flatten(), jnp.zeros((N*N*3))]),
        measurement_flag=True,
    )
    U_gate = Gate(
        gate=unitary_fun,
        initial_params = jnp.concatenate([jnp.eye(N).flatten(), jnp.zeros((N*N))]),
        measurement_flag=False,
    )

    return [decay_gate] + [povm_gate] * N_meas + [U_gate]


# Formatters
experiments_format = [ # Format of variables as tuples (name, type, required by grape, required by lut, required by rnn)
    ("t", int, True, True, True), # number timesteps (or segments)
    ("l", int, False, True, False), # LUT memory
    ("w", list, True, True, True), # reward weights
    ("Nqubits", int, True, True, True), # number of chains
    ("Nmeas", int, True, True, True), # number of measurements per timestep
    ("gammap", float, True, True, True), # sig^+ rate
    ("gammam", float, True, True, True), # sig^- rate
    ("rhot", str, True, True, True), # training state type
    ("rhoe", str, True, True, True), # evaluation state type
]

# State generation functions
state_types = {
    "discrete": generate_random_discrete_state,
    "bloch": generate_random_bloch_state,
    "excited": generate_excited_state,
    "ground": generate_ground_state,
}