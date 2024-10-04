import numpy as np


def get_points_on_grid(box_size, n_points):
    """
    Generates a grid of points within a 3D box.
    """

    n_x, n_y, n_z = n_points

    x_padding = 0.5 * box_size / (n_x + 1)
    y_padding = 0.5 * box_size / (n_y + 1)
    z_padding = 0.5 * box_size / (n_z + 1)

    x = np.linspace(x_padding, box_size - x_padding, n_x)
    y = np.linspace(y_padding, box_size - y_padding, n_y)
    z = np.linspace(z_padding, box_size - z_padding, n_z)

    X, Y, Z = np.meshgrid(x, y, z)

    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    coords = np.array(list(zip(X, Y, Z)))

    return coords
