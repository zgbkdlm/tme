"""
Generate an animated Lorenz model simulation. Almost the same as tme_lorenz.ipynb
"""
import tme.base_jax as tme
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt
from jax import jit, lax
from jax.config import config
from functools import partial
from matplotlib.animation import FuncAnimation

config.update("jax_enable_x64", True)

sigma = 10.
rho = 28.
beta = 8 / 3
Qw = jnp.eye(3)


def drift(u):
    return jnp.array([sigma * (u[1] - u[0]),
                      u[0] * (rho - u[2]) - u[1],
                      u[0] * u[1] - beta * u[2]])


bb = 0.15 * jnp.eye(3)


def dispersion(u):
    return bb


@jit
def tme_m_cov(u, dt):
    return tme.mean_and_cov(x=u, dt=dt,
                            a=drift, b=dispersion, Qw=Qw, order=3)


@jit
def em_m_cov(u, dt):
    return u + drift(u) * dt, dispersion(u) @ Qw @ dispersion(u).T * dt


@partial(jit, static_argnums=(0,))
def disc_normal(m_and_cov, x0, dts, dws):
    def scan_body(carry, elem):
        x = carry
        dt, dw = elem

        m, cov = m_and_cov(x, dt)
        chol = jnp.linalg.cholesky(cov)
        x = m + chol @ dw
        return x, x

    _, sample = lax.scan(scan_body, x0, (dts, dws))
    return sample


key = jax.random.PRNGKey(666)

# Init cond
m0 = jnp.zeros((3,))
P0 = jnp.eye(3)
key, subkey = jax.random.split(key)
x0 = jax.random.multivariate_normal(key=subkey, mean=m0, cov=P0)

# Generate ground true sample with very small dt
num_time_steps = 10000
T = jnp.linspace(0.0001, 1, num_time_steps)
dts = jnp.diff(T)
key, subkey = jax.random.split(key)
dws = jax.random.normal(key, shape=(dts.size, x0.shape[0]))

true_sample = disc_normal(em_m_cov, x0, dts, dws)

# Now make samples from EM and TME with large dt
factor = 100
T_small = T[::factor]
dts_small = dts[::factor] * factor
dws_small = dws[::factor]

sample_tme = disc_normal(tme_m_cov, x0, dts_small, dws_small)
sample_em = disc_normal(em_m_cov, x0, dts_small, dws_small)


def abs_err(x1, x2):
    return jnp.sum(jnp.sum(jnp.abs(x1 - x2)))


abs_err_tme = abs_err(true_sample[::factor], sample_tme)
abs_err_em = abs_err(true_sample[::factor], sample_em)
print(f'Cummalative abs error of TME: {abs_err_tme}')
print(f'Cummalative abs error of EM: {abs_err_em}')


def test_time():
    tic_tme = time.time()
    _ = disc_normal(tme_m_cov, x0, dts_small, dws_small)
    toc_tme = time.time()
    tic_em = time.time()
    _ = disc_normal(em_m_cov, x0, dts_small, dws_small)
    toc_em = time.time()
    return toc_tme - tic_tme, toc_em - tic_em


time_tme, time_em = test_time()

# Make true sample size consistent with em and TME for anime plot.
true_sample = true_sample[::100]

# Plot
fig = plt.figure()
ax = plt.axes(projection='3d')


def anime_init():
    ax.plot3D(true_sample[1, 0], true_sample[1, 1], true_sample[1, 2],
              c='black', linestyle='--', marker='x', label='True sample')
    ax.plot3D(sample_tme[1, 0], sample_tme[1, 1], sample_tme[1, 2],
              c='tab:blue', label=f'TME-3 sample |abs. error {abs_err_tme:.1f}| |time elapse {time_tme:.1E} s|')
    ax.plot3D(sample_em[1, 0], sample_em[1, 1], sample_em[1, 2],
              c='tab:orange', label=f'EM sample |abs. error {abs_err_em:.1f}| |time elapse {time_em:.1E} s|')
    ax.legend(loc='upper left')
    ax.set_title('TME-3 vs Euler-Maruyama (EM) on discretising \n a stochastic Lorenz model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def anime_func(frame):
    ax.plot3D(true_sample[:frame, 0], true_sample[:frame, 1], true_sample[:frame, 2],
              c='black', linestyle='--',
              marker='x', markevery=10, markersize=8)
    ax.plot3D(sample_tme[:frame, 0], sample_tme[:frame, 1], sample_tme[:frame, 2],
              c='tab:blue', label=f'TME-3 sample (abs. error ~ {abs_err_tme:.1f})')
    ax.plot3D(sample_em[:frame, 0], sample_em[:frame, 1], sample_em[:frame, 2],
              c='tab:orange', label=f'EM sample (abs. error ~ {abs_err_em:.1f})')


ani = FuncAnimation(fig, anime_func,
                    frames=100, init_func=anime_init, interval=50,
                    repeat=False)
fig.tight_layout()
fig.subplots_adjust(top=0.908, bottom=0)
plt.show()
# ani.save('lorenz_anime.gif')
