"""
Test TME disc
"""
import jax
import sympy as sp
import jax.numpy as jnp
import tme.base_sympy as tme_sp
import tme.base_jax as tme_jax
import numpy.testing as npt
from sympy import Symbol, MatrixSymbol, Matrix, sin, cos, eye, exp, simplify, tanh

jax.config.update('jax_enable_x64', True)


def test_benes():
    """For a Benes SDE we have analytical solution.
    """
    x = MatrixSymbol('x', 1, 1)

    a = Matrix([tanh(x[0])])
    b = eye(1)

    dt = Symbol('dt', positive=True)

    m, sigma = tme_sp.mean_and_cov(x, a, b, dt, order=3, simp=True)

    # Note that there is a typo in the paper.
    # \\phi_{x, 2} after equation 23 should multiply by 2.
    true_a = Matrix([x[0] + tanh(x[0]) * dt])
    true_sigma = Matrix([dt + (1 - tanh(x[0]) ** 2) * dt ** 2])

    assert (simplify(m - true_a) == Matrix([0]))
    assert (simplify(sigma - true_sigma) == Matrix([0]))


def test_generators():
    """Test vs hand-derived solutions.
    """
    dim_x = 3

    x = MatrixSymbol('x', dim_x, 1)
    a = Matrix([x[0] ** 2 + x[1],
                x[0] * x[1],
                sin(x[2])])
    b = Matrix([[x[0], 0, 0],
                [0, x[1], x[1]],
                [0, 0, x[2]]])

    # Scalar case
    phi = x[1] * x[2]
    true_result = x[0] * x[1] * x[2] + x[1] * sin(x[2]) + x[1] * x[2]
    assert (simplify(tme_sp.generator(phi, x, a, b) - true_result) == 0)

    # Vector case
    phi = Matrix([[x[1] * x[2]],
                  [exp(x[0]) + sin(x[1] + x[2])]])
    true_result = Matrix([[x[0] * x[1] * x[2] + x[1] * sin(x[2]) + x[1] * x[2]],
                          [
                              exp(x[0]) * (x[0] ** 2 + x[1]) + cos(x[1] + x[2]) * (x[0] * x[1] + sin(x[2]))
                              + 0.5 * (
                                      exp(x[0]) * x[0] ** 2 - sin(x[1] + x[2]) * (
                                      2 * x[1] ** 2 + x[2] ** 2 + 2 * x[1] * x[2])
                              )
                          ]])
    assert (simplify(tme_sp.generator_vec(phi, x, a, b) - true_result) == Matrix([[0], [0]]))

    # Matrix case
    phi = Matrix([[x[1], x[2]],
                  [x[0], sin(x[1])]])
    true_result = Matrix([[x[0] * x[1], sin(x[2])],
                          [x[0] ** 2 + x[1], cos(x[1]) * x[0] * x[1] - sin(x[1]) * x[1] ** 2]])
    assert (simplify(tme_sp.generator_mat(phi, x, a, b) - true_result) == Matrix([[0, 0], [0, 0]]))

    # Test generator powers
    order = 2

    phi = Matrix([[x[0]],
                  [x[2]]])

    true_result = [phi, Matrix([[x[0] ** 2 + x[1]],
                                [sin(x[2])]]), Matrix([[2 * x[0] * (x[0] ** 2 + x[1]) + x[0] * x[1] + x[0] ** 2],
                                                       [cos(x[2]) * sin(x[2]) - 0.5 * sin(x[2]) * x[2] ** 2]])]
    results = tme_sp.generator_power(phi, x, a, b, order)

    for i, j in zip(true_result, results):
        assert (simplify(i - j) == Matrix([[0], [0]]))


def test_vs_jax():
    """Test if the results of sympy and jax are the same.
    """
    dim_x = 2

    sym_x = sp.MatrixSymbol('x', dim_x, 1)
    sym_dt = sp.Symbol('dt', positive=True)
    a = sp.Matrix([sym_x[0] ** 2 + sym_x[1],
                   sp.sin(sym_x[1])])
    b = sp.Matrix([[1., 2., 3.],
                   [1., 1., 1.]])
    order = 4

    def phi_sp(x):
        return sp.Matrix([[x[0] * x[1], x[1] ** 3]])

    def phi_jax(x, _):
        return jnp.array([x[0] * x[1], x[1] ** 3])

    def drift(z, _):
        return jnp.array([z[0] ** 2 + z[1],
                          jnp.sin(z[1])])

    def dispersion(z, _):
        return jnp.array([[1., 2., 3.],
                          [1., 1., 1.]])

    # Test generators one by one
    list_of_Ap_sympy = tme_sp.generator_power(phi_sp(sym_x), sym_x, a, b, order)
    list_of_Ap_jax = tme_jax.generator_power(phi_jax, drift, dispersion, order)

    key = jax.random.PRNGKey(666)
    x = 0.1 * jax.random.normal(key, (dim_x,))
    for Ap_sympy, Ap_jax in zip(list_of_Ap_sympy, list_of_Ap_jax):
        Ap_func_sympy = sp.lambdify([sym_x], Ap_sympy, 'numpy')
        Ap_result_sympy = Ap_func_sympy(x.reshape(dim_x, 1))

        Ap_result_jax = Ap_jax(x, 1.)

        npt.assert_allclose(Ap_result_jax, jnp.squeeze(Ap_result_sympy))

    # Test mean and cov
    dt = 0.01
    for order in [2, 3, 4]:
        m_sympy, cov_sympy = tme_sp.mean_and_cov(sym_x, a, b, sym_dt, order, simp=True)
        m_sympy_func = sp.lambdify([sym_x, sym_dt], m_sympy, 'numpy')
        cov_sympy_func = sp.lambdify([sym_x, sym_dt], cov_sympy, 'numpy')

        m_result_sympy = m_sympy_func(x.reshape(dim_x, 1), dt)
        cov_result_sympy = cov_sympy_func(x.reshape(dim_x, 1), dt)

        @jax.jit
        def jitted_mcov(z):
            return tme_jax.mean_and_cov(z, dt, 
                                        lambda y: drift(y, None), 
                                        lambda y: dispersion(y, None), order)

        m_result_jax, cov_result_jax = jitted_mcov(x)

        npt.assert_allclose(m_result_jax, jnp.squeeze(m_result_sympy))
        npt.assert_allclose(cov_result_jax, jnp.squeeze(cov_result_sympy))
        
    # Test expectation
    for order in [2, 3, 4]:
        expec_sympy = tme_sp.expectation(phi_sp(sym_x), sym_x, a, b, sym_dt,
                                         order, simp=False)
        expec_sympy_func = sp.lambdify([sym_x, sym_dt], expec_sympy, 'numpy')
        expec_result_sympy = expec_sympy_func(x.reshape(dim_x, 1), dt)

        @jax.jit
        def jitted_expec(z):
            return tme_jax.expectation(phi_jax, z, 0., dt, drift, dispersion, order)

        expec_result_jax = jitted_expec(x)
        npt.assert_allclose(expec_result_jax, jnp.squeeze(expec_result_sympy))
