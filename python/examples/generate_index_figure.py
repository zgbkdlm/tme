"""
Generate the animated figure in the index page.
"""
import numpy as np
import jax.numpy as jnp

import tme.base_jax as tme

import matplotlib.pyplot as plt

from jax import jit
from matplotlib.animation import FuncAnimation
from typing import Tuple

alp = 1.
q = 0.1
Qw = q * jnp.eye(1)


def drift(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([x[1],
                      x[0] * (alp - x[0] ** 2) - x[1]])


# Keep in mind that we need the dispersion output a matrix
def dispersion(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([[0.],
                      [x[0]]])


@jit
def tme_m_cov(x: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return tme.mean_and_cov(x=x, dt=dt,
                            a=drift, b=dispersion, Qw=Qw,
                            order=3)


# Initial value at t=0
x = jnp.array([0., -1])

# Time instances
num_time_steps = 100
T = np.linspace(0.01, 1, num_time_steps)

# Result containers
m_results = np.zeros((num_time_steps, 2))
cov_results = np.zeros((num_time_steps, 2, 2))

# Compute for t=0.01, ..., 1
for idx, t in enumerate(T):
    m_results[idx], cov_results[idx] = tme_m_cov(x, t)


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
