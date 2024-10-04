import jax.numpy as jnp
from jax_md import space

# import numpy as np

# Helper to abstract np and jnp calculations


def calculate_distance_matrix(positions: jnp.array, box_size) -> jnp.array:
    displacement_fn, _ = space.periodic(box_size)
    pairwise_displacement_fn = space.map_product(displacement_fn)
    displacements = pairwise_displacement_fn(positions, positions)

    return space.distance(displacements)


def calculate_scattering_length_matrix(
    n_molecules, species_scattering_length: jnp.array
):
    mat_base = jnp.tile(
        species_scattering_length,
        (len(species_scattering_length) * n_molecules, n_molecules),
    )
    scattering_length_matrix = mat_base.T * mat_base

    return scattering_length_matrix
