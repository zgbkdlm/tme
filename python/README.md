# Taylor moment expansion (TME) in Python

Please see the documentation of the package in https://tme.readthedocs.io.

# Install

Install via `pip install tme` or `python setup.py install`.

# Examples

```python
import tme.base_jax as tme
import jax.numpy as jnp
from jax import jit

# Define SDE coefficients.
alp = 1.
def drift(x):
    return jnp.array([x[1],
                      x[0] * (alp - x[0] ** 2) - x[1]])


def dispersion(x):
    return jnp.array([[0.],
                      [x[0]]])


# Jit the 3-order TME mean and cov approximation functions
@jit
def tme_m_cov(x, dt):
    return tme.mean_and_cov(x=x, dt=dt,
                            a=drift, b=dispersion, Qw=jnp.eye(1),
                            order=3)

# Compute E[X(t) | X(0)=x0]
x0 = jnp.array([0., -1])
t = 1.

m_t, cov_t = tme_m_cov(x0, t)
```

Inside folder `examples`, there are a few Jupyter notebooks showing how to use the TME method (in SymPy and JaX).

# License

The GNU General Public License v3 or later
