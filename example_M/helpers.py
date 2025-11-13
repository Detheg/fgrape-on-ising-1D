# ruff: noqa
import sys, os
sys.path.append(os.path.abspath("./../feedback-grape"))
sys.path.append(os.path.abspath("./../"))

# ruff: noqa
from feedback_grape.fgrape import optimize_pulse, Decay, Gate, evaluate_on_longer_time # type: ignore
from feedback_grape.utils.states import basis # type: ignore
from feedback_grape.utils.tensor import tensor # type: ignore
from feedback_grape.utils.operators import sigmap, sigmam, identity # type: ignore
import jax.numpy as jnp
import jax
from feedback_grape.utils.fidelity import ket2dm # type: ignore
from library.utils.qubit_chain_1D import embed
from jax.scipy.linalg import expm
from tqdm import tqdm
import numpy as np

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

def generate_special_unitary(params, dim):
    assert len(params) == dim**2 - 1, "Number of real parameters must be dim^2 - 1 for an NxN special unitary matrix."
    
    H = generate_traceless_hermitian(params, dim)
    return jax.scipy.linalg.expm(-1j * H)
generate_special_unitary = jax.jit(generate_special_unitary, static_argnames=['dim'])

def generate_povm1(measurement_outcome, params):
    """ 
        Generate a 2-outcome POVM elements M_0 and M_1 for a qubit system.
        This function should parametrize all such POVMs up to unitary equivalence, i.e., M_i -> U M_i for some unitary U.
        I.e it parametrizes all pairs (M_0, M_1) such that M_0 M_0† + M_1 M_1† = I.

        measurement_outcome: 0 or 1, indicating which POVM element to generate.
        params: list of 4 real parameters [phi, theta, alpha, beta].

        when measurement_outcome == 1:
            M_1 = S D S†
        when measurement_outcome == -1:
            M_0 = S (I - D) S†

        phi, theta parametrize the unitary S, and alpha, beta parametrize the eigenvalues of M_1.
    """
    phi, theta, alpha, beta = params
    S = jnp.array(
        [[jnp.cos(phi),                   -jnp.sin(phi)*jnp.exp(-1j*theta)],
         [jnp.sin(phi)*jnp.exp(1j*theta),  jnp.cos(phi)                  ]]
    )
    s1 = jnp.sin(alpha)**2
    s2 = jnp.sin(beta)**2
    D_0 = jnp.array(
        [[s1, 0],
         [0,  s2]]
    )
    D_1 = jnp.array(
        [[(1 - s1*s1)**0.5, 0],
         [0, (1 - s2*s2)**0.5]]
    )

    return jnp.where(measurement_outcome == 1,
        tensor(identity(2), S @ D_0 @ S.conj().T),
        tensor(identity(2), S @ D_1 @ S.conj().T)
    )

def generate_povm2(measurement_outcome, params, dim):
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
    d_vec = 1e-6 + (1 - 2e-6) * d_vec # Avoid exactly 0 or 1 eigenvalues

    return jnp.where(measurement_outcome == 1,
        S @ jnp.diag(d_vec) @ S.conj().T,
        S @ jnp.diag(jnp.sqrt(1 - d_vec**2)) @ S.conj().T
    )
generate_povm2 = jax.jit(generate_povm2, static_argnames=['dim'])

def generate_decay_superoperator(N_chains, gamma):
    # 1. Construct Kraus operators for single qubit decay
    c1 = jnp.exp(-gamma/2)
    c2 = jnp.sqrt(1 - c1*c1)
    k1 = jnp.array([[1, 0],
                    [0, c1]])
    k2 = jnp.array([[0, c2],
                    [0, 0]])
    
    # 2. Construct Kraus operators for N_chains qubits
    #.   by going through all combinations of k1 and k2 for each qubit
    K = []
    for i in range(2**N_chains):
        ops = []
        for j in range(N_chains):
            if (i >> j) & 1:
                ops.append(k2)
            else:
                ops.append(k1)
        K_i = tensor(*ops)
        K_i = np.array(K_i) # Numpy arrays because they will be constant throughout (JAX should not recalculate them each time)
        K.append(K_i)

    # 3. Define function for state transition under decay
    decay_fun = lambda rho: sum([K_i @ rho @ K_i.conj().T for K_i in K])

    return jax.jit(decay_fun)

# Functions which initialize gates
def init_decay_gate(N_chains, gamma):
    decay_gate = Decay(
        c_ops = [sum([gamma * embed(sigmam(), idx, N_chains) for idx in range(N_chains)])], # dissipation on all qubits
    )

    return decay_gate

def init_decay_gate_analytical(N_chains, gamma):
    # Same as init_decay_gate, but with analytical solution for faster simulation
    decay_fun = generate_decay_superoperator(N_chains, gamma)

    decay_gate = Gate(
        gate=lambda rho, _: decay_fun(rho),
        initial_params = jnp.array([]), # No parameters
        measurement_flag = False,
        quantum_channel_flag = True,
    )

    return decay_gate

def init_povm_gate(key, N_chains):
    base_dim = 2**N_chains
    N_povm_params = base_dim*(base_dim+1)
    povm_gate = Gate(
        gate=lambda msmt, params: generate_povm2(msmt, params, dim=base_dim),
        initial_params = jax.random.uniform(key, (N_povm_params,), minval=0.0, maxval=2*jnp.pi),
        measurement_flag = True
    )

    return povm_gate

def init_unitary_gate(key, N_chains):
    base_dim = 2**N_chains
    N_unitary_params = base_dim**2
    U_gate = Gate(
        gate=lambda params: generate_unitary(params, dim=base_dim),
        initial_params = jax.random.uniform(key, (N_unitary_params,), minval=0.0, maxval=1.0),
        measurement_flag = False
    )

    return U_gate

# Functions which initialize gate combinations for the protocols
def init_simple_protocol(N_chains, gamma):
    decay_gate = init_decay_gate_analytical(N_chains, gamma)

    return [decay_gate]

def init_grape_protocol(key, N_chains, gamma):
    subkey1, subkey2 = jax.random.split(key, 2)

    decay_gate = init_decay_gate_analytical(N_chains, gamma)
    U_gate = init_unitary_gate(subkey1, N_chains)

    return [decay_gate, U_gate]

def init_fgrape_protocol(key, N_chains, gamma):
    subkey1, subkey2 = jax.random.split(key, 2)

    decay_gate = init_decay_gate_analytical(N_chains, gamma)
    povm_gate = init_povm_gate(subkey1, N_chains)
    U_gate = init_unitary_gate(subkey2, N_chains)

    return [decay_gate, povm_gate, povm_gate, povm_gate, U_gate]

# Function to generate random states
def generate_random_bloch_state(key, N_chains, noise_level):
    """
        Generate a random density matrix on Bloch sphere (cos(theta/2)*|00 ... 0> + sin(theta/2)*exp(i*phi)*sin(theta/2)*|11 ... 1>)
        plus noise term eps * identity where eps is drawn from uniform distribution between 0 and +noise_level
        and theta, phi are drawn from uniform distributions.
    """
    
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    theta = jax.random.uniform(subkey1, minval=0.0, maxval=jnp.pi)
    phi = jax.random.uniform(subkey2, minval=0.0, maxval=2*jnp.pi)
    alpha = jnp.cos(theta / 2)
    beta = jnp.sin(theta / 2) * jnp.exp(1j * phi)

    psi_one  = basis(2**N_chains, 0)
    psi_zero = basis(2**N_chains, 2**N_chains - 1)

    psi = (
        alpha * psi_zero
        + beta * psi_one
    )
    
    rho_noise = jnp.diag(jax.random.uniform(subkey3, minval=0, maxval=noise_level, shape=(2**N_chains,)))

    if noise_level == float("inf"):
        return identity(2**N_chains) / (2**N_chains)
    elif noise_level == 0:
        return ket2dm(psi)
    else:
        rho_out = ket2dm(psi) + rho_noise
        return rho_out / jnp.trace(rho_out)
generate_random_bloch_state = jax.jit(generate_random_bloch_state, static_argnames=['N_chains','noise_level'])

def generate_random_discrete_state(key, N_chains, noise_level):
    """
        Generate a random density matrix which is either |00 .. 0><00 .. 0| or |11 .. 1><11 .. 1|
        plus noise term eps * identity where eps is drawn from uniform distribution between 0 and +noise_level.
    """

    subkey1, _, subkey3 = jax.random.split(key, 3) # 3 keys for consistency with generate_random_bloch_state

    psi_one  = basis(2**N_chains, 0)
    psi_zero = basis(2**N_chains, 2**N_chains - 1)

    psi = jnp.where(jax.random.uniform(subkey1) < 0.5,
        psi_zero,
        psi_one
    )

    rho_noise = jnp.diag(jax.random.uniform(subkey3, minval=0, maxval=noise_level, shape=(2**N_chains,)))

    if noise_level == float("inf"):
        return identity(2**N_chains) / (2**N_chains)
    elif noise_level == 0:
        return ket2dm(psi)
    else:
        rho_out = ket2dm(psi) + rho_noise
        return rho_out / jnp.trace(rho_out)
generate_random_discrete_state = jax.jit(generate_random_discrete_state, static_argnames=['N_chains','noise_level'])

# Tests for the implementations
def test_implementations():
    # Test random state generation
    for callab in [lambda key, N_chains: generate_random_discrete_state(key, N_chains, noise_level=1.0), lambda key, N_chains: generate_random_bloch_state(key, N_chains, noise_level=1.0)]:
        for i in range(10):
            key = jax.random.PRNGKey(i)
            rho = callab(key, N_chains=2)

            assert jnp.allclose(rho, rho.conj().T), "Generated state is not Hermitian"
            assert jnp.isclose(jnp.trace(rho), 1.0), "Generated state does not have trace 1"
            eigvals = jnp.linalg.eigvals(rho)
            assert jnp.all(eigvals >= -1e-10), "Generated state is not positive semidefinite"


    # Test unitary and special unitary generators
    for i in range(10):
        key = jax.random.PRNGKey(i)
        key, subkey1, subkey2 = jax.random.split(key, 3)

        params = jax.random.uniform(subkey1, (16,), minval=0.0, maxval=2*jnp.pi)

        U = generate_unitary(params, 4)
        SU = generate_special_unitary(params[:-1], 4)

        assert jnp.allclose(U @ U.conj().T, jnp.eye(4)), "Unitary condition failed"
        assert jnp.allclose(SU @ SU.conj().T, jnp.eye(4)), "Special Unitary condition failed"
        assert jnp.isclose(jnp.linalg.det(SU), 1.0), "Determinant condition for Special Unitary failed"

    # Test povm generator
    for i in range(10):
        key = jax.random.PRNGKey(i)
        key, subkey = jax.random.split(key, 2)

        for f,N_params in [(generate_povm1, 4), (lambda msmt, params: generate_povm2(msmt, params, 4), 4*(4+1))]:
            params = jax.random.uniform(subkey, (N_params,), minval=0.0, maxval=2*jnp.pi)

            M_0 = f(-1, params)
            M_1 = f(1, params)

            assert jnp.allclose(M_0 @ M_0.conj().T + M_1 @ M_1.conj().T, jnp.eye(4)), "POVM elements do not sum to identity"
            assert jnp.allclose(M_0, M_0.conj().T), "POVM element M_0 is not Hermitian"
            assert jnp.allclose(M_1, M_1.conj().T), "POVM element M_1 is not Hermitian"
            assert jnp.all(jnp.linalg.eigvals(M_0) >= 0), "POVM element M_0 is not positive semidefinite"
            assert jnp.all(jnp.linalg.eigvals(M_1) >= 0), "POVM element M_1 is not positive semidefinite"