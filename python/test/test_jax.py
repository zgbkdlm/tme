import jax
import jax.numpy as jnp
import numpy.testing as npt
import tme.base_jax as tme

jax.config.update('jax_enable_x64', True)

sdes = [(lambda x, t: t * jnp.sin(x),
         lambda x, t: jnp.exp(t) * jnp.array([[x[0], 0.],
                                              [0., x[1]]])),
        (lambda x, t: -x,
         lambda x, t: jnp.eye(2))]


def euler_maruyama(key, x0, t0, T, nsteps, drift, dispersion):
    def scan_body(carry, elem):
        x = carry
        t, rnd = elem

        x = x + drift(x, t) * dt + jnp.sqrt(dt) * dispersion(x, t) @ rnd
        return x, x

    rnds = jax.random.normal(key, (nsteps, *x0.shape))
    dt = (T - t0) / nsteps
    ts = jnp.linspace(t0, T, nsteps + 1)
    xT, xs = jax.lax.scan(scan_body, x0, (ts[:-1], rnds))
    return xT, xs


def test_generator():
    """Test a single generator vs handwritten result.
    """

    drift, dispersion = sdes[0]

    def phi(x, t):
        return t * jnp.outer(x, x)

    def truth(x, t):
        a = drift(x, t)
        v11 = jnp.array([2 * x[0], 0])
        v12 = x[::-1]
        v22 = jnp.array([0, 2 * x[1]])
        return (jnp.outer(x, x)
                + jnp.array([[jnp.dot(v11, a), jnp.dot(v12, a)],
                             [jnp.dot(v12, a), jnp.dot(v22, a)]]) * t
                + jnp.exp(2 * t) * jnp.diag(x ** 2) * t)

    def actual(x, t):
        return tme.generator(phi, drift, dispersion)(x, t)

    x_ = jnp.array([1.2, 0.3])
    t_ = 0.4
    npt.assert_allclose(actual(x_, t_), truth(x_, t_))


def test_expectation_monte_carlo():
    drift, dispersion = sdes[0]

    def phi(x, t):
        return jnp.tanh(t * x)

    x0 = jnp.array([1.2, 0.3])
    t0 = 0.
    T = 0.2

    def mc_simulator(key_):
        xT = euler_maruyama(key_, x0, t0, T, 1000, drift, dispersion)[0]
        return phi(xT, T)

    key = jax.random.PRNGKey(666)
    keys = jax.random.split(key, num=100000)

    mc_result = jnp.mean(jax.vmap(mc_simulator)(keys), axis=0)
    tme_result = tme.expectation(phi, x0, t0, T, drift, dispersion, order=2)
    npt.assert_allclose(mc_result, tme_result, rtol=3e-2)


def test_expectation_lti():
    """Test generator powers vs Monte Carlo approximations.
    """

    drift, dispersion = sdes[1]

    def phi(x, t):
        return x * t

    x0 = jnp.array([1., 2.])
    t0 = 0.
    T = 1.
    true_mean = jnp.exp(-(T - t0)) * x0 * T
    approx_mean = tme.expectation(phi, x0, t0, T, drift, dispersion, order=5)
    npt.assert_allclose(approx_mean, true_mean, rtol=2e-2)


def test_mean_and_cov_lti():
    drift, dispersion = sdes[1]

    x0 = jnp.array([1., 2.])
    t0 = 0.
    T = 1.

    true_mean = jnp.exp(-(T - t0)) * x0
    true_cov = 0.5 * (1 - jnp.exp(-2 * (T - t0))) * jnp.eye(2)

    approx_m, approx_cov = tme.mean_and_cov(x0, T - t0,
                                            lambda x: drift(x, t0),
                                            lambda x: dispersion(x, t0), order=5)

    npt.assert_allclose(approx_m, true_mean, rtol=2e-2)
    npt.assert_allclose(approx_cov, true_cov, rtol=2e-2)


def test_mean_and_cov_vs_euler_maruyama():
    """TME with order=1 is consistent with Euler--Maruyama.
    """
    sigma = 10.
    rho = 28.
    beta = 8 / 3

    def drift(u):
        return jnp.array([sigma * (u[1] - u[0]),
                          u[0] * (rho - u[2]) - u[1],
                          u[0] * u[1] - beta * u[2]])

    def dispersion(u):
        return jnp.diag(jnp.array([1., u[1] * u[2], u[0]]))

    def tme_m_cov(u, dt):
        return tme.mean_and_cov(x=u, dt=dt, drift=drift, dispersion=dispersion, order=1)

    def em_m_cov(u, dt):
        return u + drift(u) * dt, dispersion(u) @ dispersion(u).T * dt

    key = jax.random.PRNGKey(666)
    x0 = jax.random.normal(key, (3,))
    T = 1.

    tme_m, tme_cov = tme_m_cov(x0, T)
    em_m, em_cov = em_m_cov(x0, T)

    npt.assert_allclose(tme_m, em_m, atol=1e-12)
    npt.assert_allclose(tme_cov, em_cov, atol=1e-12)
