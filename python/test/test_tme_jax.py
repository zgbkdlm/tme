import unittest

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import sympy as sp
import tme.base_jax as tme_jax
import tme.base_sympy as tme_sp
from jax import jit
from jax.config import config

config.update("jax_enable_x64", True)


def phi_sym(x):
    return sp.Matrix([[x[0] * x[1], x[1] ** 3]])


def phi_jax(x):
    return jnp.array([x[0] * x[1], x[1] ** 3])


def phi_jax_2d(x):
    return jnp.array([[x[0] * x[1], x[1] ** 3]])


class TestJaxVsSymPy(unittest.TestCase):
    """
    Test against SymPy results which should give true results.
    """

    def setUp(self) -> None:
        np.random.seed(111)

        self.dim_x = 2
        self.dim_w = 3

        self.sym_x = sp.MatrixSymbol('x', self.dim_x, 1)
        self.sym_dt = sp.Symbol('dt', positive=True)

        self.a = sp.Matrix([self.sym_x[0] ** 2 + self.sym_x[1],
                            sp.sin(self.sym_x[1])])

        self.b = sp.Matrix([[1., 2., 3.],
                            [1., 1., 1.]])

        self.Q = sp.eye(self.dim_w)

        self.order = 4

    def gen_Ap_sympy(self):
        return tme_sp.generator_power(phi_sym(self.sym_x),
                                      self.sym_x, self.a, self.b, self.Q, self.order)

    @staticmethod
    def a_jax(z):
        return jnp.array([z[0] ** 2 + z[1],
                          jnp.sin(z[1])])

    @staticmethod
    def b_jax(z):
        return jnp.array([[1., 2., 3.],
                          [1., 1., 1.]])

    def test_power(self):
        """Test power of generator (both generator_power() and generator_power_naive()).
        """
        x = 0.1 * np.random.randn(self.dim_x)

        list_of_Ap_sympy = self.gen_Ap_sympy()
        list_of_Ap_jax = tme_jax.generator_power(phi_jax, self.a_jax, self.b_jax, jnp.eye(self.dim_w), self.order)
        list_of_Ap_jax_naive = tme_jax.generator_power_naive(phi_jax, self.a_jax, self.b_jax, jnp.eye(self.dim_w),
                                                             self.order)

        for Ap_sympy, Ap_jax, Ap_jax_naive in zip(list_of_Ap_sympy, list_of_Ap_jax, list_of_Ap_jax_naive):
            Ap_func_sympy = sp.lambdify([self.sym_x], Ap_sympy, 'numpy')
            Ap_result_sympy = Ap_func_sympy(x.reshape(self.dim_x, 1))

            @jit
            def jitted_Ap(z):
                return Ap_jax(z)

            @jit
            def jitted_Ap_naive(z):
                return Ap_jax_naive(z)

            Ap_result_jax = jitted_Ap(jnp.array(x))
            Ap_result_jax_naive = jitted_Ap_naive(jnp.array(x))

            # Shape must be consistent with phi
            assert (phi_jax(jnp.array(x)).shape == Ap_result_jax.shape)
            assert (Ap_result_jax.shape == Ap_result_jax_naive.shape)

            # Check numerical results
            npt.assert_allclose(jnp.squeeze(Ap_result_jax), np.squeeze(Ap_result_sympy))
            npt.assert_allclose(Ap_result_jax, Ap_result_jax_naive)

    def test_mean_and_cov(self):
        """Test mean and cov approximations.
        """
        x = 0.1 * np.random.randn(self.dim_x)
        dt = 0.01

        for order in [2, 3, 4]:
            print(f'Testing order {order} for mean_and_cov().')
            # TODO: SymPy simplify() throws weird "__new__ missing" error for order >= 4.
            m_sympy, cov_sympy = tme_sp.mean_and_cov(self.sym_x, self.a, self.b, self.Q, self.sym_dt,
                                                     order, simp=False)
            m_sympy_func = sp.lambdify([self.sym_x, self.sym_dt], m_sympy, 'numpy')
            cov_sympy_func = sp.lambdify([self.sym_x, self.sym_dt], cov_sympy, 'numpy')

            m_result_sympy = m_sympy_func(x.reshape(self.dim_x, 1), dt)
            cov_result_sympy = cov_sympy_func(x.reshape(self.dim_x, 1), dt)

            @jit
            def jitted_mcov(z):
                return tme_jax.mean_and_cov(z, dt, self.a_jax, self.b_jax, jnp.eye(self.dim_w), order)

            m_result_jax, cov_result_jax = jitted_mcov(jnp.array(x))

            npt.assert_allclose(m_result_jax, np.squeeze(m_result_sympy))
            npt.assert_allclose(cov_result_jax, np.squeeze(cov_result_sympy))

    def test_expectation(self):
        """Test expectation computations.
        """
        x = 0.1 * np.random.randn(self.dim_x)
        dt = 0.01

        for order in [2, 3, 4]:
            print(f'Testing order {order} for expectation()')
            expec_sympy = tme_sp.expectation(phi_sym(self.sym_x),
                                             self.sym_x, self.a, self.b, self.Q, self.sym_dt,
                                             order, simp=False)
            expec_sympy_func = sp.lambdify([self.sym_x, self.sym_dt], expec_sympy, 'numpy')

            expec_result_sympy = expec_sympy_func(x.reshape(self.dim_x, 1), dt)

            for phi_func in [phi_jax, phi_jax_2d]:
                @jit
                def jitted_expec(z):
                    return tme_jax.expectation(phi_func, z, dt, self.a_jax, self.b_jax, jnp.eye(self.dim_w), order)

                expec_result_jax = jitted_expec(jnp.array(x))

                # Shape must be consistent with phi
                assert (phi_func(jnp.array(x)).shape == expec_result_jax.shape)

                npt.assert_allclose(jnp.squeeze(expec_result_jax),
                                    np.squeeze(expec_result_sympy))


class TestAgainstLinearSDE():
    """Test on linear SDEs which have explict mean and cov solutions.
    """
    pass
