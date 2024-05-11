import unittest
import numpy as np
from scipy.spatial import Delaunay
from tin import interpolate_tin, _interpolate_linear, _interpolate_cubic, _calculate_partial_derivatives, _calculate_cubic_coefficients, _evaluate_cubic_interpolant

class TestTINInterpolation(unittest.TestCase):
    def test_interpolate_tin_linear(self):
        points = [(0, 0, 1), (1, 0, 2), (0, 1, 3), (1, 1, 4)]
        interpolated_values = interpolate_tin(points, method='linear')

        expected_shape = (100, 100)
        self.assertEqual(interpolated_values.shape, expected_shape)

        expected_min_value = 1.0
        expected_max_value = 4.0
        self.assertAlmostEqual(np.min(interpolated_values), expected_min_value)
        self.assertAlmostEqual(np.max(interpolated_values), expected_max_value)

    def test_interpolate_tin_cubic(self):
        points = [(0, 0, 1), (1, 0, 2), (0, 1, 3), (1, 1, 4)]
        interpolated_values = interpolate_tin(points, method='cubic')

        expected_shape = (100, 100)
        self.assertEqual(interpolated_values.shape, expected_shape)

        expected_min_value = 1.0
        expected_max_value = 4.0
        self.assertAlmostEqual(np.min(interpolated_values), expected_min_value, places=1)
        self.assertAlmostEqual(np.max(interpolated_values), expected_max_value, places=1)

    def test_interpolate_tin_invalid_method(self):
        points = [(0, 0, 1), (1, 0, 2), (0, 1, 3), (1, 1, 4)]
        with self.assertRaises(ValueError):
            interpolate_tin(points, method='invalid')

    def test_interpolate_linear(self):
        x = [0, 1, 0, 1]
        y = [0, 0, 1, 1]
        z = [1, 2, 3, 4]
        tri = Delaunay(np.array([x, y]).T)
        x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))

        interpolated_values = _interpolate_linear(tri, x, y, z, x_grid, y_grid)

        expected_values = np.array([[1., 1.5, 2.],
                                    [2., 2.5, 3.],
                                    [3., 3.5, 4.]])
        np.testing.assert_array_almost_equal(interpolated_values, expected_values)

    def test_interpolate_cubic(self):
        x = [0, 1, 0, 1]
        y = [0, 0, 1, 1]
        z = [1, 2, 3, 4]
        tri = Delaunay(np.array([x, y]).T)
        x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))

        dx, dy = _calculate_partial_derivatives(tri, x, y, z)
        interpolated_values = _interpolate_cubic(tri, x, y, z, x_grid, y_grid)

        expected_values = np.array([[1.0, 1.5, 2.0],
                                    [2.0, 2.5, 3.0],
                                    [3.0, 3.5, 4.0]])
        np.testing.assert_array_almost_equal(interpolated_values, expected_values, decimal=1)

    def test_calculate_partial_derivatives(self):
        x = [0, 1, 0, 1]
        y = [0, 0, 1, 1]
        z = [1, 2, 3, 4]
        tri = Delaunay(np.array([x, y]).T)

        dx, dy = _calculate_partial_derivatives(tri, x, y, z)

        expected_dx = np.array([1, 1, 1, 1])
        expected_dy = np.array([1, 1, 1, 1])
        np.testing.assert_array_almost_equal(dx, expected_dx)
        np.testing.assert_array_almost_equal(dy, expected_dy)

    def test_calculate_cubic_coefficients(self):
        x = np.array([0, 1, 0])
        y = np.array([0, 0, 1])
        z = np.array([1, 2, 3])
        dx = np.array([1, 1, 1])
        dy = np.array([2, 2, 2])

        coefficients = _calculate_cubic_coefficients(x, y, z, dx, dy)

        expected_coefficients = np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 1])
        np.testing.assert_array_almost_equal(coefficients, expected_coefficients)

    def test_evaluate_cubic_interpolant(self):
        coefficients = np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 1])

        interpolated_value = _evaluate_cubic_interpolant(0.5, 0.5, coefficients)

        expected_value = 2.5
        self.assertAlmostEqual(interpolated_value, expected_value)

if __name__ == '__main__':
    unittest.main()