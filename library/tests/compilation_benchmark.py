# Tests how fast the function compilation is by calling fgrape.optimize_pulse on single iteration
# ruff: noqa
import sys, os
sys.path.append(os.path.abspath("./feedback-grape"))

# ruff: noqa
from feedback_grape.fgrape import optimize_pulse, Decay, Gate # type: ignore
from feedback_grape.utils.operators import cosm, sinm, identity # type: ignore
from feedback_grape.utils.states import coherent # type: ignore
import jax.numpy as jnp
import numpy as np
import jax
from feedback_grape.utils.operators import create, destroy # type: ignore
from feedback_grape.utils.fidelity import ket2dm # type: ignore
from time import time

jax.config.update("jax_enable_x64", True)

# Physical parameters
N_cav = 10  # number of cavity modes
N_snap = 5

alpha = 2
psi_target = coherent(N_cav, alpha) + coherent(N_cav, -alpha)

# Normalize psi_target before constructing rho_target
psi_target = psi_target / jnp.linalg.norm(psi_target)

rho_target = ket2dm(psi_target)

# Test system from example H
num_time_steps = 4
lut_depth = 2
reward_weights = [0, 0, 0, 1]
N_training_iterations = 20 # Number of training iterations
learning_rate = 0.01 # Learning rate
convergence_threshold = 1e-6 # Convergence threshold for early stopping

def displacement_gate(alphas):
    """Displacement operator for a coherent state."""
    alpha_re, alpha_im = alphas
    alpha = alpha_re + 1j * alpha_im
    gate = jax.scipy.linalg.expm(
        alpha * create(N_cav) - alpha.conj() * destroy(N_cav)
    )
    return gate

def initialize_displacement_gate(key):
    return Gate(
        gate=displacement_gate,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=False,
    )

def displacement_gate_dag(alphas):
    """Displacement operator for a coherent state."""
    return displacement_gate(alphas).conj().T

def initialize_displacement_gate_dag(key):
    return Gate(
        gate=displacement_gate_dag,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=False,
    )

def snap_gate(phase_list):
    diags = jnp.ones(shape=(N_cav - len(phase_list)))
    exponentiated = jnp.exp(1j * jnp.array(phase_list))
    diags = jnp.concatenate((exponentiated, diags))
    return jnp.diag(diags)

def initialize_snap_gate(key):
    return Gate(
        gate=snap_gate,
        initial_params=jax.random.uniform(
            key,
            shape=(N_snap,),
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=False,
    )

def povm_measure_operator(measurement_outcome, params):
    """
    POVM for the measurement of the cavity state.
    returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
    """
    gamma, delta = params
    cav_operator = gamma * create(N_cav) @ destroy(N_cav) + delta * identity(N_cav) / 2
    angle = cav_operator
    meas_op = jnp.where(
        measurement_outcome == 1,
        cosm(angle),
        sinm(angle),
    )
    return meas_op

def initialize_povm_gate(key):
    return Gate(
        gate=povm_measure_operator,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),  # 2 for gamma and delta
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=True,
    )

decay_gate = Decay(c_ops=[jnp.sqrt(0.005) * destroy(N_cav)])

def initialize_system_params(key):
    keys = jax.random.split(key, 4)
    return [
        decay_gate,
        initialize_povm_gate(keys[0]),
        decay_gate,
        initialize_displacement_gate(keys[1]),
        initialize_snap_gate(keys[2]),
        initialize_displacement_gate_dag(keys[3])
    ]

system_params = initialize_system_params(jax.random.PRNGKey(0))

start_time = time()
result = optimize_pulse(
    U_0=rho_target,
    C_target=rho_target,
    system_params=system_params,
    num_time_steps=num_time_steps,
    lut_depth=lut_depth,
    reward_weights=reward_weights,
    mode="lookup",
    goal="fidelity",
    max_iter=N_training_iterations,
    convergence_threshold=convergence_threshold,
    learning_rate=learning_rate,
    evo_type="density",
    batch_size=16,
)
end_time = time()
print(f"Time for single training iteration: {end_time - start_time} seconds")
print(f"Fidelity {result.final_fidelity}")

start_time = time()
result = optimize_pulse(
    U_0=rho_target,
    C_target=rho_target,
    system_params=system_params,
    num_time_steps=num_time_steps,
    lut_depth=lut_depth,
    reward_weights=reward_weights,
    mode="lookup",
    goal="fidelity",
    max_iter=N_training_iterations,
    convergence_threshold=convergence_threshold,
    learning_rate=learning_rate,
    evo_type="density",
    batch_size=16,
)
end_time = time()
print(f"Time for {N_training_iterations} training iterations: {end_time - start_time} seconds")
print(f"Fidelity {result.final_fidelity}")