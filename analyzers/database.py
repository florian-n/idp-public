import numpy as np
import os.path

FILENAME = None


def set_filename(filename: str):
    """
    Set the filename for the database adapter.
    """

    assert filename is not None, "Filename must not be None."
    assert filename.endswith(".npz"), "Filename must end with '.npz'."

    global FILENAME
    FILENAME = filename


def load_result():
    """
    Load the results from the database. If the file does not exist, returns empty arrays.
    """

    assert FILENAME is not None, "Filename must be set before loading results."
    if os.path.isfile(FILENAME) == False:
        return np.array([[]]), np.array([[]])

    data = np.load(FILENAME, allow_pickle=True)
    keys = data["keys"]
    values = np.reshape(data["values"], (len(keys), -1))
    return keys, values


def get_existing_keys():
    """
    Get the existig keys from the database
    """

    assert FILENAME is not None, "Filename must be set before loading results."
    keys, _ = load_result()
    return keys


def save_results(keys, values):
    """
    Save the results to the database.
    """

    assert FILENAME is not None, "Filename must be set before saving results."
    np.savez(FILENAME, keys=keys, values=values)


def save_intermediate_result(key: int, values: np.array):
    """
    Save teh intermediate result to the database.
    """

    assert FILENAME is not None, "Filename must be set before saving results."

    previous_keys, previous_values = load_result()
    new_keys = np.append(previous_keys, key)
    new_values = np.append(previous_values, values)
    save_results(new_keys, new_values)
