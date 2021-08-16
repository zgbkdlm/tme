"""
Taylor moment expansion (TME) in JaX.

For math details, please see the docstring of :py:mod:`tme.base_sympy`.

Functions
---------
:py:func:`generator`
    Infinitesimal generator. This is a helper function around :py:func:`generator_power`.
:py:func:`generator_power`
    Iterations/power of infinitesimal generators.
:py:func:`mean_and_cov`
    TME approximation for mean and covariance. In case you just want to compute the mean, use function
    :py:func:`expectation` with argument :code:`phi` fed by an identity function.
:py:func:`expectation`
    TME approximation for any expectation of the form :math:`\mathbb{E}[\phi(X(t + \Delta t)) \mid X(t)]`.

References
----------
See the docstring of :py:mod:`tme.base_sympy`.

Authors
-------
Adrien Corenflos and Zheng Zhao, 2021
"""

from math import factorial, comb
from typing import Callable, List, Tuple

import jax.numpy as jnp
from jax import jvp, linearize, vmap

__all__ = ['generator',
           'generator_power',
           'generator_power_naive',
           'mean_and_cov',
           'expectation']


def generator_power(phi: Callable, a: Callable, b: Callable, Qw: jnp.ndarray,
                    order: int = 1) -> List[Callable]:
    r"""Iterations/power of infinitesimal generator.

    For math details, see the docstring of :py:func:`tme.base_sympy.generator_power`.

    This is a better optimised implementation compared to :py:func:`generator_power_naive`.

    Parameters
    ----------
    phi : Callable (d,) -> (m, n)
        Target function.
    a : Callable (d,) -> (d, )
        SDE drift coefficient.
    b : Callable (d,) -> (d, w)
        SDE dispersion coefficient.
    Qw : jnp.ndarray (w, w)
        Symbolic spectral density of :math:`W`. Please note that we only tested the code when
        :code:`Qw` a constant matrix.
    order : int, optional
        Number of generator iterations. Must be >=0. Default is 1, which corresponds to the standard infinitesimal
        generator.

    Returns
    -------
    List[Callable]
        List of generator functions in ascending power order. Formally, this function returns
        :math:`[\phi, \mathcal{A}\phi, \ldots, \mathcal{A}^p\phi]`, where :code:`p` is the order.
        Each callable function in this list has exactly the save input-output shape
        signature as phi:  (d,) -> (m, n).

    Notes
    -----
    The implementation is due to Adrien Corenflos. Thank you for contributing this.
    """

    def jac_part(z, f):
        # This computes the Jacobian-vector product J[f](z) * a(z)
        return jvp(f, (z,), (a(z),))[1]

    def hess_prod_1(z, f):
        # This computes the Jacobian-vector product J[f](z) * b(z)
        _out, linearized_f = linearize(f, z)
        return vmap(linearized_f, in_axes=1, out_axes=0)(b(z))

    def hess_prod_2(z, f):
        # This computes the double Jacobian-vector product J[z -> J[f](z) * b(z)](z) * b(z).T
        # This is an equivalent, but more efficient way, to computing the Hessian form b(z).T * H[f](z) * b(z)
        # TODO: Verify is linearize is faster than vectorized vjp here.
        temp = lambda zz: hess_prod_1(zz, f)
        _out, linearized_f = linearize(temp, z)
        return vmap(linearized_f, in_axes=0, out_axes=1)(b(z).T)

    def hess_part(z, f):
        # This computes the trace of the matrix product batched along the trailing dimensions of the Hessian.
        return jnp.einsum("ii...,ii", hess_prod_2(z, f), Qw)

    gen_power = phi

    list_of_gen_powers = [gen_power]

    for _ in range(order):
        def gen_power(z, f=gen_power):
            return jac_part(z, f) + 0.5 * hess_part(z, f)

        list_of_gen_powers.append(gen_power)

    return list_of_gen_powers


def generator(phi: Callable, a: Callable, b: Callable, Qw: jnp.ndarray) -> Callable:
    r"""Infinitesimal generator for diffusion processes in Ito's SDE constructions.

    .. math::

        (\mathcal{A}\phi)(x) = \sum^d_{i=1} a_i(x)\,\frac{\partial \phi}{\partial x_i}(x)
        + \frac{1}{2}\, \sum^d_{i,j=1} \Gamma_{ij}(x) \, \frac{\partial^2 \phi}{\partial x_i \, \partial x_j}(x),

    where :math:`\phi\colon \mathbb{R}^d \to \mathbb{R}` must be sufficiently smooth function depending on the
    expansion order, and :math:`\Gamma(x) = b(x) \, b(x)^\top`.

    This is a helper function around :py:func:`generator_power`.

    Parameters
    ----------
    phi : Callable (d,) -> (m, n)
        Target function.
    a : Callable (d,) -> (d,)
        SDE drift coefficient.
    b : Callable (d,) -> (d, w)
        SDE dispersion coefficient.
    Qw : jnp.ndarray (w, w)
        Symbolic spectral density of :math:`W`. Please note that we only tested the code when
        :code:`Qw` a constant matrix.

    Returns
    -------
    Callable
        :math:`x \mapsto (\mathcal{A}\phi)(x)`.
    """
    return generator_power(phi, a, b, Qw, 1)[1]


def mean_and_cov(x: jnp.ndarray, dt: float,
                 a: Callable, b: Callable, Qw: jnp.ndarray,
                 order: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""TME approximation for mean and covariance.

    For math details, see the docstring of :py:func:`tme.base_sympy.mean_and_cov`.

    Parameters
    ----------
    x : jnp.ndarray (d,)
        The state at which the generator is evaluated. (i.e., the :math:`x` in
        :math:`\mathbb{E}[X(t + \Delta t) \mid X(t)=x]` and :math:`\mathrm{Cov}[X(t + \Delta t) \mid X(t)=x]`).
    dt : float
        Time interval.
    a : Callable (d,) -> (d,)
        SDE drift coefficient.
    b : Callable (d,) -> (d, w)
        SDE dispersion coefficient.
    Qw : jnp.ndarray (w, w)
        Symbolic spectral density of :math:`W`.
    order : int, default=3
        Order of TME. Must be >= 1.

    Returns
    -------
    m : jnp.ndarray (d,)
        TME approximation of mean :math:`\mathbb{E}[X(t + \Delta t) \mid X(t)=x]`.
    cov : jnp.ndarray (d, d)
        TME approximation of covariance :math:`\mathrm{Cov}[X(t + \Delta t) \mid X(t)=x]`.
    """
    # Give generator powers of phi^I and phi^II then evaluate them all
    list_of_A_phi_i = generator_power(lambda z: z, a=a, b=b, Qw=Qw, order=order)
    list_of_A_phi_ii = generator_power(lambda z: jnp.outer(z, z), a=a, b=b, Qw=Qw, order=order)

    A_phi_i_powers = [func(x) for func in list_of_A_phi_i]
    A_phi_ii_powers = [func(x) for func in list_of_A_phi_ii]

    # Give the mean approximation
    m = x
    for r in range(1, order + 1):
        m += 1 / factorial(r) * A_phi_i_powers[r] * dt ** r

    # Give the cov approximation
    # r = 1
    cov = A_phi_ii_powers[1] - jnp.outer(A_phi_i_powers[0], A_phi_i_powers[1]) \
          - jnp.outer(A_phi_i_powers[1], A_phi_i_powers[0])
    cov = cov * dt

    for r in range(2, order + 1):
        coeff = A_phi_ii_powers[r]
        for k in range(r + 1):
            coeff -= comb(r, k) * jnp.outer(A_phi_i_powers[k], A_phi_i_powers[r - k])
        cov += 1 / factorial(r) * coeff * dt ** r

    return m, cov


def expectation(phi: Callable,
                x: jnp.ndarray, dt: float,
                a: Callable, b: Callable, Qw: jnp.ndarray,
                order: int = 3) -> jnp.ndarray:
    r"""TME approximation of expectation on any target function :math:`\phi`.

    For math details, see the docstring of :py:func:`tme.base_sympy.expectation`.

    Parameters
    ----------
    phi : Callable (d,) -> (...)
        Target function (must be sufficiently smooth depending on the order).
    x : jnp.ndarray (d, )
        The state at which the generator is evaluated (i.e., the :math:`x` in
        :math:`\mathbb{E}[\phi(X(t + \Delta t)) \mid X(t)=x]`).
    dt : float
        Time interval.
    a : Callable (d,) -> (d, )
        SDE drift coefficient.
    b : Callable (d,) -> (d, w)
        SDE dispersion coefficient.
    Qw : jnp.ndarray (w, w)
        Symbolic spectral density of :math:`W`.
    order : int
        TME order. Must be >=0. For the relationship between the expansion order and SDE coefficient smoothness, see,
        Zhao (2021).

    Returns
    -------
    jnp.ndarray (m, n)
        TME approximation of :math:`\mathbb{E}[\phi(X(t + \Delta t)) \mid X(t)]`.
    """
    list_of_A_phi = generator_power(phi, a=a, b=b, Qw=Qw, order=order)
    Aphi = phi(x)
    for r in range(1, order + 1):
        Aphi += 1 / factorial(r) * list_of_A_phi[r](x) * dt ** r

    return Aphi
