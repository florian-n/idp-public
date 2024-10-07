import jax.numpy as jnp
from jax_md import space


def calculate_distance_matrix(positions: jnp.array, box_size) -> jnp.array:
    """
    Calculates the distance matrix between all particles r_i and r_j.
    """
    displacement_fn, _ = space.periodic(box_size)
    pairwise_displacement_fn = space.map_product(displacement_fn)
    displacements = pairwise_displacement_fn(positions, positions)

    return space.distance(displacements)


def calculate_scattering_length_matrix(
    n_molecules, species_scattering_length: jnp.array
):
    """
    Calculates a matrix of all combinations f_i and f_j.
    """
    mat_base = jnp.tile(
        species_scattering_length,
        (len(species_scattering_length) * n_molecules, n_molecules),
    )
    scattering_length_matrix = mat_base.T * mat_base

    return scattering_length_matrix
