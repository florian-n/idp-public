import numpy as np
import jax.numpy as jnp


def L1_loss(a: jnp.array, b: jnp.array) -> float:
    """
    Calculate the MEAN (!) L1 loss between two arrays (independend of size). Also ignores the first 6.67% of the data.
    """

    assert a.shape == b.shape, "Arrays must have the same shape."
    assert len(a.shape) == 1, "Arrays must be 1-dimensonal."

    skip = int(len(a) * (2 / 30))
    return jnp.mean(jnp.abs(a[skip:] - b[skip:]))


def get_losses_from_diffraction_patterns(
    patterns: np.array, ground_truth: np.array
) -> np.array:
    """
    Calculate the L1 loss between the diffraction patterns and the ground truth.
    """

    _, N_Q = patterns.shape
    N_Q_GT = ground_truth.shape[0]

    assert (
        N_Q == N_Q_GT
    ), "The number of Q-values must be the same for the patterns and the ground truth."

    return np.array([L1_loss(pattern, ground_truth) for pattern in patterns])


def get_loss_jitter(losses: np.array) -> float:
    """
    Calculate the jitter of the losses.
    """
    m, x = np.polyfit(range(len(losses)), losses, 1)
    regression_norm = m * np.arange(len(losses)) + x

    return np.std(losses - regression_norm)
