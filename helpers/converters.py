import numpy as np


def shrink_scale_box_size(original_box_size, original_n_particles, new_n_particles):
    scale_factor = (new_n_particles / original_n_particles) ** (1 / 3)
    new_box_size = original_box_size * scale_factor
    return new_box_size


def get_density(n_atoms, box_size):
    return n_atoms / box_size**3


def get_box_length_from_density(n_atoms, density):
    """
    Calculate the length of one side of a cubic box from the number of atoms and the target density.
    """
    return np.cbrt(n_atoms / density)
