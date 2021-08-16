r"""
Taylor moment expansion (TME) in SymPy.

TME applies on stochastic differential equations (SDEs) of the form

    .. math::

        d X(t) = a(X(t))dt + b(X(t)) dW(t),

    where :math:`X \colon \mathbb{T} \to \mathbb{R}^d`, and :math:`W\colon \mathbb{T} \to \mathrm{R}^w` is a Wiener
    process having spectral density :math:`Q_w \in \mathbb{R}^{w \times w}`.
    Functions :math:`a \colon \mathbb{R}^d \to \mathbb{R}^d`
    and :math:`b \colon \mathbb{R}^d \to \mathbb{R}^{d \times w}` stand for the drift and dispersion coefficients,
    respectively. :math:`\mathbb{T} = \lbrace t\in\mathbb{R}\colon t\geq t_0 \rbrace` stands for a temporal domain.

For more detailed explanations of TME, see, Zhao (2021) or Zhao et al. (2020). Notations used in this doc-site exactly
follows from Zhao (2021).

Functions
---------
:py:func:`mean_and_cov`
    TME approximation for mean and covariance. In case you just want to compute the mean, use function
    :py:func:`expectation` with argument :code:`phi` fed by an identity function.
:py:func:`expectation`
    TME approximation for any expectation of the form :math:`\mathbb{E}[\phi(X(t + \Delta t)) \mid X(t)]`.
:py:func:`generator`
    Infinitesimal generator.
:py:func:`generator_vec`
    Infinitesimal generator extended for vector-valued target function.
:py:func:`generator_mat`
    Infinitesimal generator extended for matrix-valued target function.
:py:func:`generator_power`
    Iterations/power of infinitesimal generators.

References
----------
Zheng Zhao. State-space deep Gaussian processes.
PhD thesis, Aalto University. 2021.

Zheng Zhao, Toni Karvonen, Roland Hostettler, and Simo Särkkä.
Taylor Moment Expansion for Continuous-Discrete Gaussian Filtering.
IEEE Transactions on Automatic Control, 2020.

Didier Dacunha-Castelle and Danielle Florens-Zmirou.
Estimation of the coefficients of a diffusion from discrete observations.
Stochastics, 19(4):263–284, 1986.

Danielle Florens-Zmirou. Approximate discrete-time schemes for statistics of diffusion processes.
Statistics, 20(4):547–557, 1989.

Notes
-----
Currently, this symbolc implementation has a problem as the SymPy simplification function
:code:`sympy.simplify` is not working well. See details in the docstring of function :py:func:`mean_and_cov`.

Authors
-------
Zheng Zhao, 2020, zz@zabemon.com, https://zz.zabemon.com
"""
import warnings
from typing import Tuple, List, Union, Callable

import sympy
from sympy import Symbol, Expr, Matrix, MatrixSymbol, factorial, trace, zeros
from sympy import binomial, simplify

__all__ = ['generator',
           'generator_vec',
           'generator_mat',
           'generator_power',
           'mean_and_cov',
           'expectation']


def generator(phi: Expr,
              x: MatrixSymbol,
              a: Matrix, b: Matrix, Qw: Matrix) -> Expr:
    r"""Infinitesimal generator for diffusion processes in Ito's SDE constructions.

    .. math::

        (\mathcal{A}\phi)(x) = \sum^d_{i=1} a_i(x)\,\frac{\partial \phi}{\partial x_i}(x)
        + \frac{1}{2}\, \sum^d_{i,j=1} \Gamma_{ij}(x) \, \frac{\partial^2 \phi}{\partial x_i \, \partial x_j}(x),

    where :math:`\phi\colon \mathbb{R}^d \to \mathbb{R}` must be sufficiently smooth function depending on the
    expansion order, and :math:`\Gamma(x) = b(x) \, Q_w \, b(x)^\top`.

    Parameters
    ----------
    phi : Expr
        Scalar-valued target function.
    x : MatrixSymbol
        Symbolic state vector.
    a : Matrix
        Symbolic drift coefficient.
    b : Matrix
        Symbolic dispersion coefficient.
    Qw : Matrix
        Symbolic spectral density of :math:`W`.

    Returns
    -------
    Expr
        Symbolic :math:`(\mathcal{A}\phi)(x)`.
    """
    return Matrix([phi]).jacobian(x).dot(a) \
           + 0.5 * trace(Matrix([phi]).jacobian(x).jacobian(x) * (b * Qw * b.T))


def generator_vec(phi_vec: Matrix,
                  x: MatrixSymbol,
                  a: Matrix, b: Matrix, Qw: Matrix) -> Matrix:
    r"""Infinitesimal generator for vector-valued :math:`\phi\colon \mathbb{R}^d\to\mathbb{R}^{m}`.

    Parameters
    ----------
    phi_vec : Matrix
        Vector-valued target function.
    x : MatrixSymbol
        Symbolic state vector.
    a : Matrix
        Symbolic drift coefficient.
    b : Matrix
        Symbolic dispersion coefficient.
    Qw : Matrix
        Symbolic spectral density of :math:`W`.

    Returns
    -------
    Matrix
        Symbolic :math:`(\mathcal{A}\phi)(x)`.

    Notes
    -----
    Since we are using :code:`sympy.Matrix`, the function :py:func:`generator_mat` can exactly
    replace this function.
    """
    g = zeros(phi_vec.shape[0], phi_vec.shape[1])

    for i, row in enumerate(phi_vec):
        g[i] = generator(row, x, a, b, Qw)
    return g


def generator_mat(phi_mat: Matrix,
                  x: MatrixSymbol,
                  a: Matrix, b: Matrix, Qw: Matrix) -> Matrix:
    r"""Infinitesimal generator for matrix-valued :math:`\phi\colon \mathbb{R}^d\to\mathbb{R}^{m\times n}`.

    This function exactly corresponds to the operator :math:`\overline{\mathcal{A}}` in Zhao (2021).

    Parameters
    ----------
    phi_mat : Matrix
        Matrix-valued target function.
    x : MatrixSymbol
        Symbolic state vector.
    a : Matrix
        Symbolic drift coefficient.
    b : Matrix
        Symbolic dispersion coefficient.
    Qw : Matrix
        Symbolic spectral density of :math:`W`.

    Returns
    -------
    Matrix
        Symbolic :math:`(\mathcal{A}\phi)(x)`.
    """
    g = zeros(phi_mat.shape[0], phi_mat.shape[1])

    for i in range(phi_mat.rows):
        for j in range(phi_mat.cols):
            g[i, j] = generator(phi_mat[i, j], x, a, b, Qw)
    return g


def generator_power(phi: Matrix,
                    x: MatrixSymbol,
                    a: Matrix, b: Matrix, Qw: Matrix,
                    p: int) -> List[Matrix]:
    r"""Iterations/power of infinitesimal generator for scalar/vector/matrix-valued :math:`\phi`.

    .. math::

        (\mathcal{A}^p\phi)(x),

    where :math:`p` is the number of iterations.

    Parameters
    ----------
    phi : Matrix
        Scalar/vector/Matrix-valued target function.
    x : MatrixSymbol
        Symbolic state vector.
    a : Matrix
        Symbolic drift coefficient.
    b : Matrix
        Symbolic dispersion coefficient.
    Qw : Matrix
        Symbolic spectral density of :math:`W`.
    p : int
        Power/number of iterations.

    Returns
    -------
    List[Matrix]
        A list of iterated generators
        :math:`[(\mathcal{A}^0\phi)(x), (\mathcal{A}\phi)(x), \ldots, (\mathcal{A}^p\phi)(x)]`.
    """
    generators = [phi]

    for i in range(1, p + 1):
        phi = generator_mat(phi, x, a, b, Qw)
        generators.append(phi)
    return generators


def mean_and_cov(x: MatrixSymbol,
                 a: Matrix, b: Matrix, Qw: Matrix,
                 dt: Symbol,
                 order: int = 3, simp: Union[bool, Callable] = True) -> Tuple[Matrix, Matrix]:
    r"""TME approximation for mean and covariance.

    Formally, this function approximates

    .. math::

        \mathbb{E}[X(t + \Delta t) \mid X(t)],

        \mathrm{Cov}[X(t + \Delta t) \mid X(t)].

    See, Zhao (2021, Lemma 3.4) for details.

    Parameters
    ----------
    x : MatrixSymbol
        Symbolic state vector.
    a : Matrix
        Symbolic drift coefficient.
    b : Matrix
        Symbolic dispersion coefficient.
    Qw : Matrix
        Symbolic spectral density of :math:`W`.
    dt : Symbol
        Symbolic time interval. You can create one, for example, by :code:`dt=sympy.Symbol('dt', positive=True)`.
    order : int, default=3
        Order of TME.
    simp : bool or Callable, default=True
        Set :code:`True` to simplify the results by calling the default :code:`sympy.simplify` method.
        You can also use your own simplification method by feeding it a callable function.

        .. warning::

            SymPy does not well support symbolic simplifications. The method :code:`sympy.simplify` can be
            very slow, and sometimes even give unnecessary errors. Moreover, as TME often involves complicated
            symbolic expressions, this method can really take long time!
            However, the problem does not appear in our Matlab or Mathematica implementations which simplify
            very fast and efficient.
            SymPy explained `this issue here <https://docs.sympy.org/latest/tutorial/simplification.html>`_.

    Returns
    -------
    m : Matrix
        TME approximation of mean :math:`\mathbb{E}[X(t + \Delta t) \mid X(t)]`.
    cov : Matrix
        TME approximation of covariance :math:`\mathrm{Cov}[X(t + \Delta t) \mid X(t)]`.
    """
    if dt is None:
        dt = Symbol('dt', positive=True)

    dim_x = x.shape[0]

    # Give mean estimate
    phi = Matrix([x])
    m = Matrix([x])
    for r in range(1, order + 1):
        phi = generator_vec(phi, x, a, b, Qw)
        m = m + 1 / factorial(r) * phi * dt ** r

    # Give covariance estimate
    # Precompute powers of generator
    Ax = generator_power(Matrix([x]), x, a, b, Qw, order)
    Axx = generator_power(x * x.T, x, a, b, Qw, order)

    cov = sympy.zeros(dim_x, dim_x)
    for r in range(1, order + 1):
        coeff = Axx[r]
        for s in range(r + 1):
            coeff = coeff - binomial(r, s) * Ax[s] * Ax[r - s].T
        cov = cov + 1 / factorial(r) * coeff * dt ** r

    if callable(simp):
        return simp(m), simp(cov)
    if simp:
        # TODO: simplify() unnecessarily takes too long time. Find a way to accelerate it.
        warn_simp()
        return simplify(m), simplify(cov)
    return m, cov


def expectation(phi: Matrix,
                x: MatrixSymbol,
                a: Matrix, b: Matrix, Qw: Matrix,
                dt: Symbol = None,
                order: int = 3, simp: Union[bool, Callable] = True) -> Matrix:
    r"""TME approximation of expectation on target function :math:`\phi`.

    Formally, this function approximates

    .. math::

        \mathbb{E}[\phi(X(t+\Delta t)) \mid X(t)],

    where :math:`\phi \colon \mathbb{R}^d \to \mathbb{R}^{m\times n}` is any smooth enough function of interest.

    Parameters
    ----------
    phi : Matrix
        Target function.
    x : MatrixSymbol
        Symbolic state vector.
    a : Matrix
        Symbolic drift coefficient.
    b : Matrix
        Symbolic dispersion coefficient.
    Qw : Matrix
        Symbolic spectral density of :math:`W`.
    dt : Symbol, optional
        Symbolic time interval. If this is not specified by the user, then the function will create one.
    order : int, default=3
        Order of TME.
    simp : bool or Callable, default=True
        Set True to simplify the results by calling the default :code:`sympy.simplify` method. You can also give
        your own simplification method as a callable function input.

    Returns
    -------
    Matrix
        TME approximation of :math:`\mathbb{E}[\phi(X(t + \Delta t)) \mid X(t)]`.
    """
    if dt is None:
        dt = Symbol('dt', positive=True)

    # Precompute generator powers
    Ar = generator_power(phi, x, a, b, Qw, order)

    expec = sympy.zeros(*phi.shape)
    for r in range(order + 1):
        expec = expec + 1 / factorial(r) * Ar[r] * dt ** r

    if callable(simp):
        return simp(expec)
    if simp:
        # TODO: simplify() unnecessarily takes too long time. Find a way to accelerate it.
        # TODO: simplify() throws weird errors in some rare cases.
        warn_simp()
        return simplify(expec)
    return expec


def warn_simp():
    """Warn about sympy.simplify
    """
    warnings.warn('The simplification method sympy.simplify can be very slow and throw weird errors.')
