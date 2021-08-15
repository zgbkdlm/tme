"""
Taylor moment expansion (TME) in JaX.

For math details, please see the docstring of :py:mod:`tme.base_sympy`.

Functions
---------
:py:func:`phi_i`
    Target function for inducing mean approximation.
:py:func:`phi_ii`
    Target function for inducing second moment approximation.
:py:func:`generator`
    Infinitesimal generator.
:py:func:`generator_power`
    Iterations/power of infinitesimal generators.
:py:func:`generator_power_naive`
    A naive implementation of iterations/power of infinitesimal generators.
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
import jax.numpy as jnp

from jax import jacfwd, jvp, hessian, linearize, vmap
from math import factorial, comb
from typing import Callable, List, Tuple

__all__ = ['phi_i',
           'phi_ii',
           'generator',
           'generator_power',
           'generator_power_naive',
           'mean_and_cov',
           'expectation']


def phi_i(x: jnp.ndarray) -> jnp.ndarray:
    r"""Target function for inducing mean :math:`\mathbb{E}[X(t + \Delta t) \mid X(t)]`.

    Nothing but an identity function.

    See, Zhao 2021, Section 3.3.

    Parameters
    ----------
    x : jnp.ndarray (d, )

    Returns
    -------
    jnp.ndarray (d, )
    """
    return x


def phi_ii(x: jnp.ndarray) -> jnp.ndarray:
    r"""Target function for inducing second moment
    :math:`\mathbb{E}[X(t + \Delta t) \, X(t + \Delta t)^\top \mid X(t)]`.

    See, Zhao 2021, Section 3.3.

    Parameters
    ----------
    x : jnp.ndarray (d, )

    Returns
    -------
    jnp.ndarray (d, d)
    """
    return jnp.outer(x, x)


def generator(phi: Callable,
              x: jnp.ndarray,
              a: Callable, b: Callable, Qw: jnp.ndarray) -> jnp.ndarray:
    r"""Infinitesimal generator for diffusion processes in Ito's SDE constructions.

    For math details, see the docstring of :obj:`tme.base_sympy.generator` (or more precisely, should be
    :obj:`tme.base_sympy.mat`).

    Parameters
    ----------
    phi : Callable (d, ) -> (m, n)
        Target function.
    x : jnp.ndarray (d, )
        The state at which the generator is evaluated.
    a : Callable (d, ) -> (d, )
        SDE drift coefficient.
    b : Callable (d, ) -> (d, w)
        SDE dispersion coefficient.
    Qw : jnp.ndarray (w, w)
        Symbolic spectral density of :math:`W`.

    Returns
    -------
    jnp.ndarray (d, ) -> (m, n)
        :math:`(\mathcal{A}\phi)(x)`.

    Notes
    -----
    The generator here refers to :math:`\overline{\mathcal{A}}` in Zhao (2021). With a slight abuse of
    notation, we will keep on using :math:`\mathcal{A}` in this doc page.
    """
    bb = b(x)
    return jacfwd(phi)(x) @ a(x) + 0.5 * jnp.trace(hessian(phi)(x) @ (bb @ Qw @ bb.T), axis1=-2, axis2=-1)


def generator_power_naive(phi: Callable, phi_out_ndims: int,
                          a: Callable, b: Callable, Qw: jnp.ndarray,
                          order: int) -> List[Callable]:
    """Iterations/power of infinitesimal generator in a naive implementation.

    This function is almost the same as with :py:func:`generator_power`, except that here the code is not
    really optimised but a direct and crude implementation of generator iterations.
    For details, see :py:func:`generator_power`.

    Since the code here is extremely simple, this function could be a good backup if
    :py:func:`generator_power` somehow fails.

    Notes
    -----
    By default, functions :py:func:`mean_and_cov` and :py:func:`expectation` call :py:func:`generator_power`
    instead of this naive implementation.

    The argument :code:`phi_out_ndims` is not used but kept here in order to keep consistent with
    :py:func:`generator_power`.
    """
    list_of_gen_powers = [phi]

    gen_power = phi

    for _ in range(order):
        def gen_power(z, f=gen_power): return generator(f, z, a, b, Qw)

        list_of_gen_powers.append(gen_power)

    return list_of_gen_powers


def generator_power(phi: Callable, phi_out_ndims: int,
                    a: Callable, b: Callable, Qw: jnp.ndarray,
                    order: int) -> List[Callable]:
    r"""Iterations/power of infinitesimal generator.

    For math details, see the docstring of :py:func:`tme.base_sympy.generator_power`.

    This is a better optimised implementation compared to :py:func:`generator_power_naive`.

    Parameters
    ----------
    phi : Callable (d, ) -> (m, n)
        Target function.
    phi_out_ndims : int, default=2
        The implementation needs to transpose certain Hessian product matrices.
        Hence, the function has to know what the output shape (dim) of function :code:`phi` is.
        Specify here the number of output dimensions of :code:`phi`.
    a : Callable (d, ) -> (d, )
        SDE drift coefficient.
    b : Callable (d, ) -> (d, w)
        SDE dispersion coefficient.
    Qw : jnp.ndarray (w, w)
        Symbolic spectral density of :math:`W`. Please note that we only tested the code when
        :code:`Qw` a constant matrix.
    order : int
        Number of generator iterations. Must be >=0.

    Returns
    -------
    List[Callable]
        List of generator functions in ascending power order. Formally, this function returns
        :math:`[\phi, \mathcal{A}\phi, \ldots, \mathcal{A}^p\phi]`, where :code:`p` is the order.
        Each callable function in this list has exactly the save input-output shape
        signature as phi:  (d, ) -> (m, n).

    Notes
    -----
    The implementation is due to Adrien Corenflos. Thank you for contributing this.
    """
    if phi_out_ndims == 1:
        perm_table = [2, 0, 1]
    elif phi_out_ndims == 2:
        perm_table = [2, 3, 0, 1]
    else:
        raise ValueError(f'Output dimension {phi_out_ndims} of phi is not supported yet.')

    def jac_part(z, f):
        return jvp(f, (z,), (a(z),))[1]

    def hess_prod_1(z, f):
        _out, linearized_f = linearize(f, z)
        return vmap(linearized_f, in_axes=1, out_axes=0)(b(z))

    def hess_prod_2(z, f):
        temp = lambda zz: hess_prod_1(zz, f)
        _out, linearized_f = linearize(temp, z)
        return vmap(linearized_f, in_axes=0, out_axes=1)(b(z).T)

    def hess_part(z, f):
        hess_val = jnp.transpose(hess_prod_2(z, f), perm_table)
        res = jnp.trace(hess_val @ Qw, axis1=-2, axis2=-1)
        return res

    gen_power = phi

    list_of_gen_powers = [gen_power]

    for _ in range(order):
        def gen_power(z, f=gen_power):
            return jac_part(z, f) + 0.5 * hess_part(z, f)

        list_of_gen_powers.append(gen_power)

    return list_of_gen_powers


def mean_and_cov(x: jnp.ndarray, dt: float,
                 a: Callable, b: Callable, Qw: jnp.ndarray,
                 order: int = 3,
                 gen_pow: Callable = generator_power) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""TME approximation for mean and covariance.

    For math details, see the docstring of :py:func:`tme.base_sympy.mean_and_cov`.

    Parameters
    ----------
    x : jnp.ndarray (d, )
        The state at which the generator is evaluated. (i.e., the :math:`x` in
        :math:`\mathbb{E}[X(t + \Delta t) \mid X(t)=x]` and :math:`\mathrm{Cov}[X(t + \Delta t) \mid X(t)=x]`).
    dt : float
        Time interval.
    a : Callable (d, ) -> (d, )
        SDE drift coefficient.
    b : Callable (d, ) -> (d, w)
        SDE dispersion coefficient.
    Qw : jnp.ndarray (w, w)
        Symbolic spectral density of :math:`W`.
    order : int, default=3
        Order of TME. Must be >= 1.
    gen_pow : Callable, default=generator_power
        Callable function to compute powers of generators, Default is :py:func:`generator_power`, but you can
        replace it with the naive implementation :py:func:`generator_power_naive` if something go wrong
        with the default :py:func:`generator_power`.

    Returns
    -------
    m : jnp.ndarray (d, )
        TME approximation of mean :math:`\mathbb{E}[X(t + \Delta t) \mid X(t)=x]`.
    cov : jnp.ndarray (d, d)
        TME approximation of covariance :math:`\mathrm{Cov}[X(t + \Delta t) \mid X(t)=x]`.
    """
    # Give generator powers of phi^I and phi^II then evaluate them all
    list_of_A_phi_i = gen_pow(phi_i, phi_out_ndims=1, a=a, b=b, Qw=Qw, order=order)
    list_of_A_phi_ii = gen_pow(phi_ii, phi_out_ndims=2, a=a, b=b, Qw=Qw, order=order)

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


def expectation(phi: Callable, phi_out_ndims: int,
                x: jnp.ndarray, dt: float,
                a: Callable, b: Callable, Qw: jnp.ndarray,
                order: int = 3,
                gen_pow: Callable = generator_power) -> jnp.ndarray:
    r"""TME approximation of expectation on any target function :math:`\phi`.

    For math details, see the docstring of :py:func:`tme.base_sympy.expectation`.

    Parameters
    ----------
    phi : Callable (d, ) -> (m, n)
        Target function (must be sufficiently smooth depending on the order).
    phi_out_ndims : int
        See the docstring of :py:func:`generator_power`.
    x : jnp.ndarray (d, )
        The state at which the generator is evaluated (i.e., the :math:`x` in
        :math:`\mathbb{E}[\phi(X(t + \Delta t)) \mid X(t)=x]`).
    dt : float
        Time interval.
    a : Callable (d, ) -> (d, )
        SDE drift coefficient.
    b : Callable (d, ) -> (d, w)
        SDE dispersion coefficient.
    Qw : jnp.ndarray (w, w)
        Symbolic spectral density of :math:`W`.
    order : int
        TME order. Must be >=0. For the relationship between the expansion order and SDE coefficient smoothness, see,
        Zhao (2021).
    gen_pow : Callable, default=generator_power
        Callable function to compute :math:`\mathcal{A}^p\phi`, where p is the order.
        Default is :py:func:`generator_power`, but you can replace it with the naive implementation
        :py:func:`generator_power_naive` if something go wrong with the default :py:func:`generator_power`.

    Returns
    -------
    jnp.ndarray (m, n)
        TME approximation of :math:`\mathbb{E}[\phi(X(t + \Delta t)) \mid X(t)]`.
    """
    list_of_A_phi = gen_pow(phi, phi_out_ndims=phi_out_ndims, a=a, b=b, Qw=Qw, order=order)

    Aphi = phi(x)
    for r in range(1, order + 1):
        Aphi += 1 / factorial(r) * list_of_A_phi[r](x) * dt ** r

    return Aphi
