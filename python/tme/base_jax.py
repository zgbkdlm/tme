r"""
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
# TODO: The logic further down can be improved dramatically if we make diagonal noise specific logic.

try:
    import jax as _
except:
    raise ImportError("By default the library is not packaged with JaX due to the need to support CPU and GPU users. "
                      "In order to use it, follow the instructions on https://github.com/google/jax#installation.")

import math
import jax.numpy as jnp
from math import factorial
from jax import jvp, linearize, vmap
from typing import Callable, List, Tuple
from tme.typings import Array, JArray, FloatScalar

__all__ = ['generator',
           'generator_power',
           'mean_and_cov',
           'expectation']


def generator_power(phi: Callable[[Array, FloatScalar], JArray],
                    drift: Callable[[Array, FloatScalar], JArray], dispersion: Callable[[Array, FloatScalar], JArray],
                    order: int = 1) -> List[Callable[[Array, FloatScalar], JArray]]:
    r"""Iterations/power of infinitesimal generator.

    For math details, see the docstring of :py:func:`tme.base_sympy.generator_power`.

    Parameters
    ----------
    phi : Callable (d, float) -> (...)
        Target function.
    drift : Callable (d, float) -> (d,)
        SDE drift coefficient.
    dispersion : Callable (d, float) -> (d, w)
        SDE dispersion coefficient, where `w` stands for the dimension of the Wiener process.
    order : int, optional
        Number of generator iterations. Must be >=0. Default is 1, which corresponds to the standard infinitesimal
        generator.

    Returns
    -------
    List[Callable]
        List of generator functions in ascending power order. Formally, this function returns
        :math:`[\phi, \mathcal{A}\phi, \ldots, \mathcal{A}^p\phi]`, where :code:`p` is the order.
        Each callable function in this list has exactly the same input-output shape
        signature as phi:  (d, float) -> (...).

    Notes
    -----
    The implementation was initiated by Adrien Corenflos. Thank you for contributing this.

    You may also find a naive implementation of infinitesimal generators and their iterations in the test file
    :code:`./test/test_tme_jax.py`.
    """

    def jac_part(x, t, f):
        return jvp(f, (x, t), (drift(x, t), 1.))[1]

    def hess_prod_1(x, f, b):
        _, linearized_f = linearize(f, x)
        return vmap(linearized_f, in_axes=1, out_axes=0)(b)

    def hess_prod_2(x, t, f):
        b = _format_dispersion(dispersion(x, t))
        f_x = lambda x_: f(x_, t)
        _, linearized_f = linearize(lambda xx: hess_prod_1(xx, f_x, b), x)
        return vmap(linearized_f, in_axes=0, out_axes=1)(b.T)

    gen_power = phi

    list_of_gen_powers = [gen_power]

    for _ in range(order):
        def gen_power(x, t, f=gen_power):
            return jac_part(x, t, f) + 0.5 * jnp.einsum("ii...", hess_prod_2(x, t, f))

        list_of_gen_powers.append(gen_power)

    return list_of_gen_powers


def generator(phi: Callable[[Array, FloatScalar], JArray],
              drift: Callable[[Array, FloatScalar], JArray],
              dispersion: Callable[[Array, FloatScalar], JArray]) -> Callable[[Array, FloatScalar], JArray]:
    r"""Infinitesimal generator for diffusion processes in Ito's SDE constructions.

    .. math::

        (\mathcal{A}\phi)(x) = \sum^d_{i=1} a_i(x)\,\frac{\partial \phi}{\partial x_i}(x)
        + \frac{1}{2}\, \sum^d_{i,j=1} \Gamma_{ij}(x) \, \frac{\partial^2 \phi}{\partial x_i \, \partial x_j}(x),

    where :math:`\phi\colon \mathbb{R}^d \to \mathbb{R}` must be sufficiently smooth function depending on the
    expansion order, and :math:`\Gamma(x) = b(x) \, b(x)^\top`.

    This is a helper function around :py:func:`generator_power`.

    Parameters
    ----------
    phi : Callable (d, float) -> (...)
        Target function.
    drift : Callable (d, float) -> (d,)
        SDE drift coefficient.
    dispersion : Callable (d, float) -> (d, w)
        SDE dispersion coefficient, where `w` stands for the dimension of the Wiener process.

    Returns
    -------
    Callable (d, float) -> (...)
        A callable function which carries out :math:`x \mapsto \mathcal{A}\phi`. The output shape of this function
        is the same as :code:`phi`.
    """
    return generator_power(phi, drift, dispersion, 1)[1]


def mean_and_cov(x: Array, dt: FloatScalar,
                 drift: Callable[[Array], JArray], dispersion: Callable[[Array], JArray],
                 order: int = 3) -> Tuple[JArray, JArray]:
    r"""TME approximation for mean and covariance.

    For math details, see the docstring of :py:func:`tme.base_sympy.mean_and_cov`.

    Parameters
    ----------
    x : JArray (d,)
        The state at which the generator is evaluated. (i.e., the :math:`x` in
        :math:`\mathbb{E}[X(t + \Delta t) \mid X(t)=x]` and :math:`\mathrm{Cov}[X(t + \Delta t) \mid X(t)=x]`).
    dt : float
        Time interval.
    drift : Callable (d,) -> (d,)
        SDE drift coefficient.
    dispersion : Callable (d,) -> (d, w)
        SDE dispersion coefficient, where `w` stands for the dimension of the Wiener process.
    order : int, default=3
        Order of TME. Must be >= 1.

    Returns
    -------
    m : JArray (d,)
        TME approximation of mean :math:`\mathbb{E}[X(t + \Delta t) \mid X(t)=x]`.
    cov : JArray (d, d)
        TME approximation of covariance :math:`\mathrm{Cov}[X(t + \Delta t) \mid X(t)=x]`.

    Notes
    -----
    - When `order = 1`, the TME mean and cov approximations are exactly the same with Euler--Maruyama.
    - This function does not support time-dependent drift and dispersion coefficients. Note the signature
    difference of `drift` and `dispersion` compared to `expectation`. The reason is that it is not clear how
    a principled covariance approximation will be made.
    """
    # Give generator powers of phi^I and phi^II then evaluate them all
    list_of_Aphi_i = _generator_power_ti(lambda z: z, drift, dispersion, order)
    list_of_Aphi_ii = _generator_power_ti(lambda z: jnp.outer(z, z), drift, dispersion, order)

    Aphi_i_powers = [func(x) for func in list_of_Aphi_i]
    Aphi_ii_powers = [func(x) for func in list_of_Aphi_ii]

    # Give the mean approximation
    m = x
    for r in range(1, order + 1):
        m = m + 1 / factorial(r) * Aphi_i_powers[r] * dt ** r

    # Give the cov approximation
    # r = 1
    cov = Aphi_ii_powers[1] - jnp.outer(Aphi_i_powers[0], Aphi_i_powers[1]) \
          - jnp.outer(Aphi_i_powers[1], Aphi_i_powers[0])
    cov = cov * dt

    for r in range(2, order + 1):
        coeff = Aphi_ii_powers[r]
        for k in range(r + 1):
            coeff = coeff - _comb(r, k) * jnp.outer(Aphi_i_powers[k], Aphi_i_powers[r - k])
        cov = cov + 1 / factorial(r) * coeff * dt ** r

    return m, cov


def expectation(phi: Callable[[Array, FloatScalar], JArray],
                x: Array, t: FloatScalar, dt: FloatScalar,
                drift: Callable[[Array, FloatScalar], JArray],
                dispersion: Callable[[Array, FloatScalar], JArray],
                order: int = 3) -> JArray:
    r"""TME approximation of expectation on any target function :math:`\phi`.

    For math details, see the docstring of :py:func:`tme.base_sympy.expectation`.

    Parameters
    ----------
    phi : Callable (d, float) -> (...)
        Target function (must be sufficiently smooth depending on the order).
    x : JArray (d, )
        The state at which the generator is evaluated (i.e., the :math:`x` in
        :math:`\mathbb{E}[\phi(X(t + \Delta t)) \mid X(t)=x]`).
    t : FloatScalar
        Time at which :math:`\phi(\cdot, t)` is computed.
    dt : float
        Time interval.
    drift : Callable (d, float) -> (d,)
        SDE drift coefficient.
    dispersion : Callable (d, float) -> (d, w)
        SDE dispersion coefficient, where `w` stands for the dimension of the Wiener process.
    order : int
        Order of TME. Must be >=0. For the relationship between the expansion order and SDE coefficient smoothness, see,
        Zhao (2021).

    Returns
    -------
    JArray (...)
        TME approximation of :math:`\mathbb{E}[\phi(X(t + \Delta t), t + \Delta t) \mid X(t)]`.
        The output shape is consistence with the input shape of :code:`phi`.
    """
    list_of_Aphi = generator_power(phi, drift, dispersion, order)
    Aphi = phi(x, t)
    for r in range(1, order + 1):
        Aphi += 1 / factorial(r) * list_of_Aphi[r](x, t) * dt ** r

    return Aphi


def _generator_power_ti(phi: Callable[[Array], JArray],
                        drift: Callable[[Array], JArray],
                        dispersion: Callable[[Array], JArray],
                        order: int = 1) -> List[Callable[[Array], JArray]]:
    """The same as with :py:func:`tme.base_jax.generator_power` but works only for time-independent
    phi, drift, and dispersion.
    """

    def jac_part(x, f):
        return jvp(f, (x,), (drift(x),))[1]

    def hess_prod_1(x, f, b):
        _, linearized_f = linearize(f, x)
        return vmap(linearized_f, in_axes=1, out_axes=0)(b)

    def hess_prod_2(x, f):
        b = _format_dispersion(dispersion(x))
        _, linearized_f = linearize(lambda xx: hess_prod_1(xx, f, b), x)
        return vmap(linearized_f, in_axes=0, out_axes=1)(b.T)

    gen_power = phi

    list_of_gen_powers = [gen_power]

    for _ in range(order):
        def gen_power(x, f=gen_power):
            return jac_part(x, f) + 0.5 * jnp.einsum("ii...", hess_prod_2(x, f))

        list_of_gen_powers.append(gen_power)

    return list_of_gen_powers


def _comb(n, k):
    try:
        return math.comb(n, k)
    except AttributeError:  # Python version < 3.8 does not have math.comb
        return _manual_comb(n, k)


def _manual_comb(n, k):
    if k > n // 2:
        return _manual_comb(n, n - k)
    if k < 0:
        return 0
    if k == 0:
        return 1
    return _manual_comb(n - 1, k - 1) + _manual_comb(n - 1, k)


def _format_dispersion(bz):
    ndim = jnp.ndim(bz)
    if ndim == 0:
        return jnp.atleast_2d(bz)
    if ndim == 1:
        return jnp.expand_dims(bz, 1)
    if ndim == 2:
        return bz
    else:
        raise ValueError(f"Dispersion coefficient b(z) must have at most 2 dimensions. {ndim} were passed")
