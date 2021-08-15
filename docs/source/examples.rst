Examples
========
This section shows runnable TME examples in Python and Matlab. In particular, we use the following two example models.

1. Model 1,

	.. math::

		d X(t) = \mathrm{tanh}(X(t)) dt + dW(t).

2. Model 2,

	.. math::

		d X_1(t) &= X_2(t), \\
		d X_2(t) &= (X_1(t)\, (\kappa - (X_1(t))^2)) dt + X_1(t) dW(t).

We want to compute their mean, covariance/variance, or more generally :math:`\mathbb{E}[\phi(X(t + \Delta t)) \mid X(t)]` for any function of interest :math:`\phi`.

In Python
---------

See, `Jupyter Notebook (SymPy) for Model 1 <https://github.com/zgbkdlm/tme/blob/main/python/examples/benes_sympy.ipynb>`_.

See, `Jupyter Notebook (JaX) for Model 1 <https://github.com/zgbkdlm/tme/blob/main/python/examples/benes_jax.ipynb>`_.

See, `Jupyter Notebook (JaX) for Model 2 <https://github.com/zgbkdlm/tme/blob/main/python/examples/nonlinear_multidim_jax.ipynb>`_.

In Matlab
---------

See, `Matalb codes for Models 1 and 2 <https://github.com/zgbkdlm/tme/tree/main/matlab>`_
