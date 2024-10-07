import numpy as np


def analyze_gradients_absolute(grads: np.array):
    """
    Analyze the gradients by their absolute values.
    """

    assert grads is not None, "Gradients must not be None."
    assert len(grads.shape) == 1, "Gradients must be 1-dimensional."

    return np.mean(np.abs(grads)), np.std(np.abs(grads))


def analyze_gradients_magnitudal(grads: np.array):
    """
    Analyze the gradients by their magnitudes.
    """

    assert grads is not None, "Gradients must not be None."
    assert len(grads.shape) == 1, "Gradients must be 1-dimensional."

    magnitudes = np.log10(np.abs(grads))
    return np.mean(magnitudes), np.std(magnitudes)
