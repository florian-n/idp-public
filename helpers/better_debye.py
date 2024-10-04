# Source:
# https://pythoninchemistry.org/sim_and_scat/calculating_scattering/debye_equation.html

import jax.numpy as np
from jax import lax
import jax

from helpers.debye_helper import calculate_distance_matrix


def debye(
    qvalues: np.array, distance_matrix: np.array, scattering_length_matrix: np.array
) -> np.array:
    @jax.remat
    def for_q(carry, q):
        q_r = q * distance_matrix
        sin_q_r = np.sin(q_r)
        inner_sum = scattering_length_matrix * (sin_q_r / q_r)
        i_q = np.nansum(inner_sum)

        return carry, i_q

    _, result = lax.scan(for_q, None, qvalues)

    return np.array(result)


@jax.jit
def get_averaged_debye(
    frames: np.array,
    qvalues: np.array,
    box_size: int,
    scattering_length_matrix: np.array,
):
    @jax.remat
    def get_debye_result(carry, positions):
        distances = calculate_distance_matrix(positions, box_size)
        result = debye(qvalues, distances, scattering_length_matrix)
        return carry, result

    _, results = lax.scan(get_debye_result, None, frames)

    return results
