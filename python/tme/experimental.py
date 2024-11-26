"""
Experimental features.
"""
import jax
import jax.numpy as jnp
from jax import jacfwd, jvp
from folx import forward_laplacian, register_function, wrap_forward_laplacian
from tme.typings import Array, JArray, FloatScalar
from typing import Callable, List, Tuple


def generator_power_diagonal(phi: Callable[[Array, FloatScalar], JArray],
                             drift: Callable[[Array, FloatScalar], JArray],
                             dispersion: FloatScalar,
                             order: int = 1) -> List[Callable[[Array, FloatScalar], JArray]]:
    """When the dispersion is a constant diagonal, we can compute the Laplacian part efficiently.
    Experimental, do not use.
    """

    def jac_part(x, t, f):
        return jvp(f, (x, t), (drift(x, t), 1.))[1]

    def lap(x, t, f):
        f_x = lambda x_: f(x_, t)
        return forward_laplacian(f_x)(x)

    gen_power = phi

    list_of_gen_powers = [gen_power]

    for _ in range(order):
        def gen_power(x, t, f=gen_power):
            fwd_lap_results = lap(x, t, f)
            # return (jacfwd(f, argnums=1)(x, t)
            #         + fwd_lap_results.jacobian.dense_array @ drift(x, t)
            #         + 0.5 * fwd_lap_results.laplacian * dispersion ** 2)
            return jac_part(x, t,
                            f) + 0.5 * fwd_lap_results.laplacian * dispersion ** 2  # This will compute the Jacobian twice

        list_of_gen_powers.append(gen_power)

    return list_of_gen_powers
