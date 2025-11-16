import sys, os
sys.path.append(os.path.abspath("./../../feedback-grape"))
sys.path.append(os.path.abspath("./../../"))
sys.path.append(os.path.abspath("./../"))

import jax.numpy as jnp
from feedback_grape.utils.tensor import tensor
from feedback_grape.utils.states import basis
from feedback_grape.utils.operators import sigmax, sigmay, sigmaz

# Selection of states and operators
def negative(M):
    return jnp.sqrt(jnp.eye(M.shape[0]) - M)

N = 2**3 # Hilbert space dimension
psi_00 = basis(N, 3)
psi_01 = basis(N, 2)
psi_10 = basis(N, 1)
psi_11 = basis(N, 0)
I = jnp.eye(N)
I2 = jnp.eye(2)
O = jnp.zeros((N,N))

# Pauli operators on three qubits
Z1 = tensor(sigmaz(), I2, I2)
Z2 = tensor(I2, sigmaz(), I2)
Z3 = tensor(I2, I2, sigmaz())
X1 = tensor(sigmax(), I2, I2)
X2 = tensor(I2, sigmax(), I2)
X3 = tensor(I2, I2, sigmax())
Y1 = tensor(sigmay(), I2, I2)
Y2 = tensor(I2, sigmay(), I2)
Y3 = tensor(I2, I2, sigmay())

# Stabilizers
S1 = Z1 @ Z2
S2 = Z2 @ Z3

M1_p = (I + S1)/2
M1_m = (I - S1)/2
M2_p = (I + S2)/2
M2_m = (I - S2)/2

# Error correction protocol to test
protocol = {
    "label": "Stabilizer code",
    # povms for first time step
    "povm1_init_+": M1_p,
    "povm1_init_-": M1_m,
    "povm2_++": M2_p,
    "povm2_+-": M2_m,
    "povm2_-+": M2_p,
    "povm2_--": M2_m,
    # povms for all subsequent time steps
    "povm1_+++": M1_p,
    "povm1_++-": M1_m,
    "povm1_+-+": M1_p,
    "povm1_+--": M1_m,
    "povm1_-++": M1_p,
    "povm1_-+-": M1_m,
    "povm1_--+": M1_p,
    "povm1_---": M1_m,
    "povm2_+++": M2_p,
    "povm2_++-": M2_m,
    "povm2_+-+": M2_p,
    "povm2_+--": M2_m,
    "povm2_-++": M2_p,
    "povm2_-+-": M2_m,
    "povm2_--+": M2_p,
    "povm2_---": M2_m,
    # unitaries for all measurement outcomes
    "U_++": I,
    "U_+-": X3,
    "U_-+": X1,
    "U_--": X2,
}