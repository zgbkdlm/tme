"""
Test TME disc
"""
import unittest

import tme.base_sympy as tme
from sympy import Symbol, MatrixSymbol, Matrix, sin, cos, eye, exp, simplify, tanh


class TestGenerator(unittest.TestCase):
    """
    Test generators on scale, vec, and matrix-valued phi functions
    """

    def setUp(self) -> None:
        self.dim_x = 3
        self.dim_w = 3

        self.x = MatrixSymbol('x', self.dim_x, 1)

        self.a = Matrix([self.x[0] ** 2 + self.x[1],
                         self.x[0] * self.x[1],
                         sin(self.x[2])])

        self.b = Matrix([[self.x[0], 0, 0],
                         [0, self.x[1], self.x[1]],
                         [0, 0, self.x[2]]])

    def test_scalar(self) -> None:
        """
        Test generator()
        """
        x = self.x

        phi = x[1] * x[2]
        true_result = x[0] * x[1] * x[2] + x[1] * sin(x[2]) + x[1] * x[2]

        assert (simplify(tme.generator(phi, x, self.a, self.b) - true_result) == 0)

    def test_vector(self) -> None:
        """
        Test generator_vec()
        """
        x = self.x

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

        assert (simplify(tme.generator_vec(phi, x, self.a, self.b) - true_result) == Matrix([[0], [0]]))

    def test_matrix(self) -> None:
        """
        Test generator_mat()
        """
        x = self.x

        phi = Matrix([[x[1], x[2]],
                      [x[0], sin(x[1])]])
        true_result = Matrix([[x[0] * x[1], sin(x[2])],
                              [x[0] ** 2 + x[1], cos(x[1]) * x[0] * x[1] - sin(x[1]) * x[1] ** 2]])

        assert (simplify(tme.generator_mat(phi, x, self.a, self.b) - true_result) == Matrix([[0, 0], [0, 0]]))

    def test_power(self) -> None:
        """
        Test generator_power
        """
        x = self.x

        order = 2

        phi = Matrix([[x[0]],
                      [x[2]]])

        true_result = [phi, Matrix([[x[0] ** 2 + x[1]],
                                    [sin(x[2])]]), Matrix([[2 * x[0] * (x[0] ** 2 + x[1]) + x[0] * x[1] + x[0] ** 2],
                                                           [cos(x[2]) * sin(x[2]) - 0.5 * sin(x[2]) * x[2] ** 2]])]
        results = tme.generator_power(phi, x, self.a, self.b, order)

        for i, j in zip(true_result, results):
            assert (simplify(i - j) == Matrix([[0], [0]]))


class TestBenes(unittest.TestCase):
    """
    Test TME on a Benese model
    """

    def setUp(self) -> None:
        self.x = MatrixSymbol('x', 1, 1)

        # IMPORTANT: In scalar case, you have to use x[0] instead of x
        self.a = Matrix([tanh(self.x[0])])

        self.b = eye(1)

        self.Q = eye(1)

        self.dt = Symbol('dt', positive=True)

    def test_tme(self) -> None:
        x = self.x
        dt = self.dt

        m, sigma = tme.mean_and_cov(self.x, self.a, self.b, self.dt, order=3, simp=True)

        # Note that there is a typo in the paper.
        # \\phi_{x, 2} after equation 23 should multiply by 2.
        true_a = Matrix([x[0] + tanh(x[0]) * dt])
        true_sigma = Matrix([dt + (1 - tanh(x[0]) ** 2) * dt ** 2])

        assert (simplify(m - true_a) == Matrix([0]))
        assert (simplify(sigma - true_sigma) == Matrix([0]))


if __name__ == '__main__':
    unittest.main()
