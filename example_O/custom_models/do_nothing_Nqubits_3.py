import sys, os
sys.path.append(os.path.abspath("./../../feedback-grape"))
sys.path.append(os.path.abspath("./../../"))
sys.path.append(os.path.abspath("./../"))

import jax.numpy as jnp

# Selection of states and operators

N = 2**3 # Hilbert space dimension
I = jnp.eye(N)
O = jnp.zeros((N,N))

# Error correction protocol to test
protocol = {
    "label": "Do nothing",
    # povms for first time step
    "povm1_init_+": I / 2**0.5,
    "povm1_init_-": I / 2**0.5,
    "povm2_++": I / 2**0.5,
    "povm2_+-": I / 2**0.5,
    "povm2_-+": I / 2**0.5,
    "povm2_--": I / 2**0.5,
    # povms for all subsequent time steps
    "povm1_+++": I / 2**0.5,
    "povm1_++-": I / 2**0.5,
    "povm1_+-+": I / 2**0.5,
    "povm1_+--": I / 2**0.5,
    "povm1_-++": I / 2**0.5,
    "povm1_-+-": I / 2**0.5,
    "povm1_--+": I / 2**0.5,
    "povm1_---": I / 2**0.5,
    "povm2_+++": I / 2**0.5,
    "povm2_++-": I / 2**0.5,
    "povm2_+-+": I / 2**0.5,
    "povm2_+--": I / 2**0.5,
    "povm2_-++": I / 2**0.5,
    "povm2_-+-": I / 2**0.5,
    "povm2_--+": I / 2**0.5,
    "povm2_---": I / 2**0.5,
    # unitaries for all measurement outcomes
    "U_++": I,
    "U_+-": I,
    "U_-+": I,
    "U_--": I,
}