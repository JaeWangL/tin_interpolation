from typing import List, Tuple
import numpy as np
from scipy.spatial import Delaunay

def interpolate_tin(points: List[Tuple[float, float, float]], method: str = 'linear') -> np.ndarray:
    """
    Interpolate values using the TIN (Triangulated Irregular Network) algorithm.

    Args:
        points: A list of tuples representing the points in the format (x, y, z).
        method: The interpolation method to use. Can be either 'linear' or 'cubic'.

    Returns:
        A 2D numpy array representing the interpolated values.
    """
    # Extract x, y, and z coordinates from the points
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    # Create a Delaunay triangulation
    tri = Delaunay(np.array([x, y]).T)

    # Create a grid of points for interpolation
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Interpolate values using the chosen method
    if method == 'linear':
        interpolated_values = _interpolate_linear(tri, x, y, z, x_grid, y_grid)
    elif method == 'cubic':
        interpolated_values = _interpolate_cubic(tri, x, y, z, x_grid, y_grid)
    else:
        raise ValueError(f"Invalid interpolation method: {method}")

    return interpolated_values

def _interpolate_linear(tri: Delaunay, x: List[float], y: List[float], z: List[float],
                        x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """
    Perform linear interpolation using the TIN.

    Args:
        tri: The Delaunay triangulation object.
        x: The x-coordinates of the points.
        y: The y-coordinates of the points.
        z: The z-coordinates (values) of the points.
        x_grid: The x-coordinates of the grid points for interpolation.
        y_grid: The y-coordinates of the grid points for interpolation.

    Returns:
        A 2D numpy array representing the interpolated values.
    """
    # Find the simplices containing each grid point
    simplices = tri.find_simplex(np.array([x_grid.ravel(), y_grid.ravel()]).T)

    # Calculate the barycentric coordinates of each grid point within its simplex
    barycentric_coords = np.empty((simplices.size, 3))
    for i, simplex in enumerate(simplices):
        b = tri.transform[simplex, :2].dot(np.array([x_grid.ravel()[i], y_grid.ravel()[i]]) - tri.transform[simplex, 2])
        barycentric_coords[i] = np.array([b[0], b[1], 1 - b.sum()])

    # Interpolate the values using the barycentric coordinates
    interpolated_values = np.einsum('ij,ij->i', np.take(z, tri.simplices[simplices]), barycentric_coords)

    return interpolated_values.reshape(x_grid.shape)

def _interpolate_cubic(tri: Delaunay, x: List[float], y: List[float], z: List[float],
                       x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """
    Perform cubic interpolation using the Clough-Tocher method.

    Args:
        tri: The Delaunay triangulation object.
        x: The x-coordinates of the points.
        y: The y-coordinates of the points.
        z: The z-coordinates (values) of the points.
        x_grid: The x-coordinates of the grid points for interpolation.
        y_grid: The y-coordinates of the grid points for interpolation.

    Returns:
        A 2D numpy array representing the interpolated values.
    """
    # Find the simplices containing each grid point
    simplices = tri.find_simplex(np.array([x_grid.ravel(), y_grid.ravel()]).T)

    # Calculate the barycentric coordinates of each grid point within its simplex
    barycentric_coords = np.empty((simplices.size, 3))
    for i, simplex in enumerate(simplices):
        b = tri.transform[simplex, :2].dot(np.array([x_grid.ravel()[i], y_grid.ravel()[i]]) - tri.transform[simplex, 2])
        barycentric_coords[i] = np.array([b[0], b[1], 1 - b.sum()])

    # Calculate the partial derivatives at each vertex of the triangulation
    dx, dy = _calculate_partial_derivatives(tri, x, y, z)

    # Perform Clough-Tocher cubic interpolation
    interpolated_values = np.zeros(x_grid.size)
    for i, simplex in enumerate(simplices):
        vertices = tri.simplices[simplex]
        x_vertices = np.take(x, vertices)
        y_vertices = np.take(y, vertices)
        z_vertices = np.take(z, vertices)
        dx_vertices = np.take(dx, vertices)
        dy_vertices = np.take(dy, vertices)

        # Calculate the cubic interpolant coefficients for the current simplex
        coefficients = _calculate_cubic_coefficients(x_vertices, y_vertices, z_vertices, dx_vertices, dy_vertices)

        # Evaluate the cubic interpolant at the current grid point
        x_point, y_point = x_grid.ravel()[i], y_grid.ravel()[i]
        interpolated_values[i] = _evaluate_cubic_interpolant(x_point, y_point, coefficients)

    return interpolated_values.reshape(x_grid.shape)

def _calculate_partial_derivatives(tri: Delaunay, x: List[float], y: List[float], z: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the partial derivatives at each vertex of the triangulation.

    Args:
        tri: The Delaunay triangulation object.
        x: The x-coordinates of the points.
        y: The y-coordinates of the points.
        z: The z-coordinates (values) of the points.

    Returns:
        A tuple containing two numpy arrays representing the partial derivatives with respect to x and y.
    """
    # Calculate the partial derivatives using finite differences
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)

    for i, simplex in enumerate(tri.simplices):
        vertices = tri.points[simplex]
        z_values = np.array(z)[simplex]

        # Calculate the partial derivatives using central differences
        dx_simplex = np.zeros(3)
        dy_simplex = np.zeros(3)

        for j in range(3):
            k = (j + 1) % 3
            if vertices[k, 0] - vertices[j, 0] != 0:
                dx_simplex[j] = (z_values[k] - z_values[j]) / (vertices[k, 0] - vertices[j, 0])
            if vertices[k, 1] - vertices[j, 1] != 0:
                dy_simplex[j] = (z_values[k] - z_values[j]) / (vertices[k, 1] - vertices[j, 1])

        # Assign the average partial derivatives to the corresponding vertices
        dx[simplex] = np.mean(dx_simplex)
        dy[simplex] = np.mean(dy_simplex)

    return dx, dy

def _calculate_cubic_coefficients(x: np.ndarray, y: np.ndarray, z: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Calculate the cubic interpolant coefficients for a single simplex.

    Args:
        x: The x-coordinates of the simplex vertices.
        y: The y-coordinates of the simplex vertices.
        z: The z-coordinates (values) of the simplex vertices.
        dx: The partial derivatives with respect to x at the simplex vertices.
        dy: The partial derivatives with respect to y at the simplex vertices.

    Returns:
        A numpy array representing the cubic interpolant coefficients.
    """
    # Calculate the cubic interpolant coefficients
    coefficients = np.zeros((10,))

    # Construct the system of equations
    A = np.array([
        [x[0]**3, x[0]**2*y[0], x[0]*y[0]**2, y[0]**3, x[0]**2, x[0]*y[0], y[0]**2, x[0], y[0], 1],
        [x[1]**3, x[1]**2*y[1], x[1]*y[1]**2, y[1]**3, x[1]**2, x[1]*y[1], y[1]**2, x[1], y[1], 1],
        [x[2]**3, x[2]**2*y[2], x[2]*y[2]**2, y[2]**3, x[2]**2, x[2]*y[2], y[2]**2, x[2], y[2], 1],
        [3*x[0]**2, 2*x[0]*y[0], y[0]**2, 0, 2*x[0], y[0], 0, 1, 0, 0],
        [3*x[1]**2, 2*x[1]*y[1], y[1]**2, 0, 2*x[1], y[1], 0, 1, 0, 0],
        [3*x[2]**2, 2*x[2]*y[2], y[2]**2, 0, 2*x[2], y[2], 0, 1, 0, 0],
        [0, x[0]**2, 2*x[0]*y[0], 3*y[0]**2, 0, x[0], 2*y[0], 0, 1, 0],
        [0, x[1]**2, 2*x[1]*y[1], 3*y[1]**2, 0, x[1], 2*y[1], 0, 1, 0],
        [0, x[2]**2, 2*x[2]*y[2], 3*y[2]**2, 0, x[2], 2*y[2], 0, 1, 0],
        [6*x[0], 2*y[0], 0, 0, 2, 0, 0, 0, 0, 0]
    ])

    b = np.array([z[0], z[1], z[2], dx[0], dx[1], dx[2], dy[0], dy[1], dy[2], 0])

    # Solve the system of equations using least-squares
    coefficients = np.linalg.lstsq(A, b, rcond=None)[0]

    return coefficients

def _evaluate_cubic_interpolant(x: float, y: float, coefficients: np.ndarray) -> float:
    """
    Evaluate the cubic interpolant at a given point.

    Args:
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.
        coefficients: The cubic interpolant coefficients.

    Returns:
        The interpolated value at the given point.
    """
    # Evaluate the cubic interpolant
    interpolated_value = (
        coefficients[0]*x**3 + coefficients[1]*x**2*y + coefficients[2]*x*y**2 + coefficients[3]*y**3 +
        coefficients[4]*x**2 + coefficients[5]*x*y + coefficients[6]*y**2 +
        coefficients[7]*x + coefficients[8]*y + coefficients[9]
    )

    return interpolated_value