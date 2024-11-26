"""
Generate an animated Lorenz model simulation. Almost the same as tme_lorenz.ipynb
"""
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tme.base_jax as tme
from jax import jit, lax
from matplotlib.animation import FuncAnimation

jax.config.update("jax_enable_x64", True)

sigma = 10.
rho = 28.
beta = 8 / 3


def drift(u):
    return jnp.array([sigma * (u[1] - u[0]),
                      u[0] * (rho - u[2]) - u[1],
                      u[0] * u[1] - beta * u[2]])


bb = 0.15 * jnp.eye(3)


def dispersion(_):
    return bb


def tme_m_cov(u, dt, order):
    return tme.mean_and_cov(x=u, dt=dt,
                            drift=drift, dispersion=dispersion, order=order)


def em_m_cov(u, dt):
    return u + drift(u) * dt, dispersion(u) @ dispersion(u).T * dt


@partial(jit, static_argnums=(0,))
def disc_normal(m_and_cov, x0, dts, dws):
    def scan_body(carry, elem):
        x = carry
        dt, dw = elem

        m, cov = m_and_cov(x, dt)
        x = m + jnp.linalg.cholesky(cov) @ dw
        return x, x

    _, sample = lax.scan(scan_body, x0, (dts, dws))
    return sample


tme_m_cov_2 = jit(partial(tme_m_cov, order=2))
tme_m_cov_3 = jit(partial(tme_m_cov, order=3))
tme_m_cov_4 = jit(partial(tme_m_cov, order=4))

key = jax.random.PRNGKey(666)

# Init cond
m0 = jnp.zeros((3,))
P0 = jnp.eye(3)
key, _ = jax.random.split(key)
x0 = jax.random.multivariate_normal(key=key, mean=m0, cov=P0)

# Generate ground true sample with very small dt and high order TME
num_time_steps = 100_000_0
T = jnp.linspace(1e-5, 10, num_time_steps)
dts = jnp.diff(T)
key, _ = jax.random.split(key)
dws = jax.random.normal(key, shape=(dts.size, x0.shape[0]))

true_sample = disc_normal(tme_m_cov_4, x0, dts, dws)

# Now make samples from EM and TME with large dt
factor = 1000
T_small = T[::factor]
dts_small = dts[::factor] * factor
dws_small = dws[::factor]

sample_tme_2 = disc_normal(tme_m_cov_2, x0, dts_small, dws_small)
sample_tme_3 = disc_normal(tme_m_cov_3, x0, dts_small, dws_small)
sample_em = disc_normal(em_m_cov, x0, dts_small, dws_small)


def abs_err(x1, x2):
    return jnp.mean(jnp.sum(jnp.abs(x1 - x2), 1))


abs_err_tme_2 = abs_err(true_sample[::factor], sample_tme_2)
abs_err_tme_3 = abs_err(true_sample[::factor], sample_tme_3)
abs_err_em = abs_err(true_sample[::factor], sample_em)
print(f'Average abs error of TME order 2: {abs_err_tme_2}')
print(f'Average abs error of TME order 3: {abs_err_tme_3}')
print(f'Average abs error of EM: {abs_err_em}')


def test_time(cov_fun, dts, dws, n_iter=10):
    tic = time.time()
    for _ in range(n_iter):
        _ = disc_normal(cov_fun, x0, dts, dws).block_until_ready()
    toc = time.time()

    return (toc - tic) / n_iter


time_tme_2 = test_time(tme_m_cov_2, dts_small, dws_small)
time_tme_3 = test_time(tme_m_cov_3, dts_small, dws_small)
time_em = test_time(em_m_cov, dts_small, dws_small)

print(f'Average runtime of TME-2: {time_tme_2}')
print(f'Average runtime of TME-3: {time_tme_3}')
print(f'Average runtime of EM: {time_em}')

# Make true sample size consistent with em and TME for anime plot.
true_sample = true_sample[::factor]

# Plot
fig = plt.figure()
ax = plt.axes(projection='3d')

l1, = ax.plot3D(true_sample[1, 0], true_sample[1, 1], true_sample[1, 2],
                c='black', linestyle='--', label='True sample', marker='x', markevery=factor // 100, zorder=3)
l2, = ax.plot3D(sample_tme_2[1, 0], sample_tme_2[1, 1], sample_tme_2[1, 2],
                c='#7bccc4',
                label=f'TME-2 sample |abs. error {abs_err_tme_2:.1f}| |average runtime {time_tme_2:.1E} s|', zorder=2)
l3, = ax.plot3D(sample_tme_3[1, 0], sample_tme_3[1, 1], sample_tme_3[1, 2],
                c='#43a2ca',
                label=f'TME-3 sample |abs. error {abs_err_tme_3:.1f}| |average runtime {time_tme_3:.1E} s|', zorder=1)
l4, = ax.plot3D(sample_em[1, 0], sample_em[1, 1], sample_em[1, 2],
                c='#0868ac', label=f'EM sample |abs. error {abs_err_em:.1f}| |average runtime {time_em:.1E} s|',
                zorder=-1)

ax.legend(loc='upper left')
ax.set_title('TME-3 vs Euler-Maruyama (EM) on discretising \n a stochastic Lorenz model')
ax.set_xlim3d([-30, 30])
ax.set_xlabel('X')

ax.set_ylim3d([-30, 30])
ax.set_ylabel('Y')

ax.set_zlim3d([0, 50])
ax.set_zlabel('Z')


def anime_init():
    l1.set_data_3d(true_sample[0, 0], true_sample[0, 1], true_sample[0, 2])
    l2.set_data_3d(sample_tme_2[0, 0], sample_tme_2[0, 1], sample_tme_2[0, 2])
    l3.set_data_3d(sample_tme_3[0, 0], sample_tme_3[0, 1], sample_tme_3[0, 2])
    l4.set_data_3d(sample_em[0, 0], sample_em[0, 1], sample_em[0, 2])

    return l1, l2, l3, l4


def anime_func(frame):
    l1.set_data_3d(true_sample[:frame, 0], true_sample[:frame, 1], true_sample[:frame, 2])
    l2.set_data_3d(sample_tme_2[:frame, 0], sample_tme_2[:frame, 1], sample_tme_2[:frame, 2])
    l3.set_data_3d(sample_tme_3[:frame, 0], sample_tme_3[:frame, 1], sample_tme_3[:frame, 2])
    l4.set_data_3d(sample_em[:frame, 0], sample_em[:frame, 1], sample_em[:frame, 2])

    return l1, l2, l3, l4


ani = FuncAnimation(fig, anime_func,
                    frames=range(0, len(dts_small), 1), init_func=anime_init, interval=50,
                    repeat=False)
fig.tight_layout()
fig.subplots_adjust(top=0.908, bottom=0.01)
# plt.show()
ani.save('lorenz_anime.gif')
