"""
Convenient JAX typings.
"""
import jax
import numpy as np
from typing import Any, Union

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
Array = Union[JArray, np.ndarray]

# Scalar values
FloatScalar = Union[float, JFloat]
IntScalar = Union[int, JFloat]
BoolScalar = Union[bool, JBool]
