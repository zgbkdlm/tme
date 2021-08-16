"""
Generate the animated figure in the index page.
"""
from typing import Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tme.base_jax as tme
from jax import vmap
from matplotlib.animation import FuncAnimation

alp = 1.
Qw = 0.1    # Float Qw will be converted to a matrix in generator_power.


def drift(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([x[1],
                      x[0] * (alp - x[0] ** 2) - x[1]])


# Keep in mind that the shapes of dispersion and Qw should be consistent.
def dispersion(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([0., x[0]])


def tme_m_cov(x: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return tme.mean_and_cov(x=x, dt=dt,
                            a=drift, b=dispersion, Qw=Qw,
                            order=3)


# Initial value at t=0
x = jnp.array([0., -1])

# Time instances
num_time_steps = 100
T = np.linspace(0.01, 1, num_time_steps)

# Compute for t=0.01, ..., 1
m_results, cov_results = vmap(tme_m_cov, in_axes=[None, 0])(x, T)


def anime_init():
    plt.plot(T[:1], m_results[:1, 0],
             linewidth=3, c='tab:blue',
             label='TME-3 X_1(t)')
    plt.plot(T[:1], m_results[:1, 1],
             linewidth=3, c='tab:orange',
             label='TME-3 X_2(t)')
    plt.xlim(0, 1)
    plt.legend(loc='upper right')
    plt.title('TME-3 approximation of mean E[X(t) | X(0)=x0] \n from a Duffing-van der Pol equation.')
    plt.xlabel('t')
    plt.ylabel('E[X(t) | X(0)=x0]')


def anime_func(frame):
    plt.plot(T[:frame], m_results[:frame, 0],
             linewidth=3, c='tab:blue', label='')
    plt.plot(T[:frame], m_results[:frame, 1],
             linewidth=3, c='tab:orange', label='')
    plt.xlim(0, 1)


fig = plt.figure()

ani = FuncAnimation(fig, anime_func,
                    frames=num_time_steps, init_func=anime_init, interval=20,
                    repeat=False)

ani.save('index_tme_duffing.gif')

# plt.show()
