import jax.numpy as jnp
import numpy as np


def get_slices(all_positions: jnp.array, start, n_samples, normalized_timestep):
    """
    Slice a trajectory into n_samples. The first sample is at start. n_samples equally spaced samples are taken.
    """
    indices = np.array(
        np.floor(np.linspace(start, len(all_positions) - 1, n_samples)),
        dtype=int,
    )
    selected_states = jnp.array(all_positions[indices])
    timesteps = indices * normalized_timestep
    return selected_states, timesteps
