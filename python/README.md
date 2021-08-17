# Taylor moment expansion (TME) in Python

Please see the documentation of the package in https://tme.readthedocs.io.

# Install

Install via `pip install tme` or `python setup.py install` (Please note that if you would like to use JaX, please 
install `jax` by yourself beforehand).

# Examples

```python
import tme.base_jax as tme
import jax.numpy as jnp
from jax import vmap

# Define SDE coefficients.
alp = 1.
def drift(x):
    return jnp.array([x[1],
                      x[0] * (alp - x[0] ** 2) - x[1]])


def dispersion(x):
    return jnp.array([0., x[0]])


# Jit the 3-order TME mean and cov approximation functions
def tme_m_cov(x, dt):
    return tme.mean_and_cov(x=x, dt=dt,
                            a=drift, b=dispersion, Qw=jnp.eye(1),
                            order=3)

# Compute E[X(t) | X(0)=x0] for several time steps
x0 = jnp.array([0., -1])
ts = jnp.array([0.25, 0.5, 1.])

m_t, cov_t = vmap(tme_m_cov, in_axes=[None, 0])(x0, ts)
```

Inside folder `examples`, there are a few Jupyter notebooks showing how to use the TME method (in SymPy and JaX).

# License

The GNU General Public License v3 or later
