"""
Convenient JAX typings.
"""
import jax
import numpy as np
from typing import Any

# The three types are exactly the same alias of jax.Array. We differ them only semantically.
JArray = jax.Array
JInt = jax.Array
JFloat = jax.Array
JBool = jax.Array
JKey = jax.Array

# Pytree
# PyTree = JArray | List['PyTree'] | Dict[str, 'PyTree'] | Tuple['PyTree', ...]
PyTree = Any

# Arrays
Array = JArray | np.ndarray

# Scalar values
FloatScalar = float | JFloat
IntScalar = int | JFloat
BoolScalar = bool | JBool
