import jax
import jax.numpy as jnp
import numpy as np
import time

# Tests different implementations of matrix vector multiplication
# where the vector is mostly zeros except for one element set to 1.0
size = 100
boolean = True

def multiply(mat, vec1, vec2):
    if boolean:
        return vec1.conj().T @ mat @ vec2
    else:
        return 0

def multiply_efficient(mat):
    if boolean:
        return mat.at[0, 0].get()
    else:
        return 0

mat = jax.random.uniform(jax.random.PRNGKey(0), (size, size)) + 1j * jax.random.uniform(jax.random.PRNGKey(1), (size, size))
vec1 = jnp.zeros(size)
vec1 = vec1.at[0].set(1.0).T
vec2 = jnp.zeros(size)
vec2 = vec2.at[0].set(1.0).T

# Test function with was not jitted
start = time.time()

for i in range(10000):
    multiply(mat, vec1, vec2)

end = time.time()
print(f"Time taken: {end - start} seconds")

# Test function with jitted multiply
multiply1 = jax.jit(multiply)
start = time.time()

for i in range(10000):
    multiply1(mat, vec1, vec2)

end = time.time()
print(f"Time taken: {end - start} seconds")

# Test function with jitted lambda
multiply2 = jax.jit(lambda m: multiply(m, vec1, vec2))
start = time.time()

for i in range(10000):
    multiply2(mat)

end = time.time()
print(f"Time taken: {end - start} seconds")

# Test jitted efficient function
multiply_efficient = jax.jit(multiply_efficient)
start = time.time()

for i in range(10000):
    multiply_efficient(mat)

end = time.time()
print(f"Time taken: {end - start} seconds")

# Try again with numpy arrays as vectors -> less overhead from jax arrays in first test, otherwise almost no difference
def multiply(mat, vec1, vec2):
    if boolean:
        return vec1.conj().T @ mat @ vec2
    else:
        return 0

def multiply_efficient(mat):
    if boolean:
        return mat.at[0, 0].get()
    else:
        return 0

mat = jax.random.uniform(jax.random.PRNGKey(0), (size, size)) + 1j * jax.random.uniform(jax.random.PRNGKey(1), (size, size))
vec1 = np.zeros(size)
vec1[0] = 1.0
vec1 = vec1.T
vec2 = np.zeros(size)
vec2[0] = 1.0
vec2 = vec2.T

# Test function with was not jitted
start = time.time()

for i in range(10000):
    multiply(mat, vec1, vec2)

end = time.time()
print(f"Time taken: {end - start} seconds")

# Test function with jitted multiply
multiply1 = jax.jit(multiply)
start = time.time()

for i in range(10000):
    multiply1(mat, vec1, vec2)

end = time.time()
print(f"Time taken: {end - start} seconds")

# Test function with jitted lambda
multiply2 = jax.jit(lambda m: multiply(m, vec1, vec2))
start = time.time()

for i in range(10000):
    multiply2(mat)

end = time.time()
print(f"Time taken: {end - start} seconds")

# Test jitted efficient function
multiply_efficient = jax.jit(multiply_efficient)
start = time.time()

for i in range(10000):
    multiply_efficient(mat)

end = time.time()
print(f"Time taken: {end - start} seconds")
