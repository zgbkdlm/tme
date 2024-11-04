"""
Experimental features.
"""
import jax
import jax.numpy as jnp
from jax import jacfwd
from folx import forward_laplacian
from typing import Callable, List, Tuple

from tme.typings import Array, JArray, FloatScalar


def generator_power_diagonal(phi: Callable[[Array, FloatScalar], JArray],
                             drift: Callable[[Array, FloatScalar], JArray],
                             dispersion: FloatScalar,
                             order: int = 1) -> List[Callable[[Array, FloatScalar], JArray]]:
    """When the dispersion is a constant diagonal.
    """

    def lap(x, t, f):
        f_x = lambda x_: f(x_, t)
        return forward_laplacian(f_x)(x)

    gen_power = phi

    list_of_gen_powers = [gen_power]

    for _ in range(order):
        def gen_power(x, t, f=gen_power):
            fwd_lap_results = lap(x, t, f)
            return (jacfwd(f, argnums=1)(x, t)
                    + fwd_lap_results.jacobian.dense_array @ drift(x, t)
                    + 0.5 * fwd_lap_results.laplacian * dispersion ** 2)

        list_of_gen_powers.append(gen_power)

    return list_of_gen_powers
