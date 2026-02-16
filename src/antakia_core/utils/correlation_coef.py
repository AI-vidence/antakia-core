import numpy as np


def has_duplicates(array):
    # Convert the array to a 1D numpy array if it's not already
    array = np.ravel(array)

    # Sort the array
    sorted_array = np.sort(array)

    # Check if any element is equal to its neighboring element
    duplicates = np.any(sorted_array[:-1] == sorted_array[1:])

    return duplicates


def ranks_of_array(array):
    # Get the indices that would sort the array
    sorted_indices = np.argsort(array)

    # Create an array to hold the ranks
    ranks = np.empty_like(sorted_indices)

    # Fill the ranks array with the corresponding ranks
    ranks[sorted_indices] = np.arange(len(array))

    return ranks


def sort_array_by_reference(target_array, reference_array):
    # Get the indices that would sort the reference array
    sorted_indices = np.argsort(reference_array)

    # Sort the target array based on the sorted indices of the reference array
    sorted_array = target_array[sorted_indices]

    return sorted_array


def shuffle_dataframe_and_series(dataframe, series):
    # Shuffle the index
    shuffled_index = np.random.permutation(dataframe.index)

    # Shuffle the DataFrame using the shuffled index
    shuffled_dataframe = dataframe.reindex(shuffled_index).reset_index(
        drop=True)

    # Shuffle the Series using the shuffled index
    shuffled_series = series.reindex(shuffled_index).reset_index(drop=True)

    return shuffled_dataframe, shuffled_series


def correlation_coef(x: np.ndarray, y: np.ndarray):
    """
    implementation de https://arxiv.org/pdf/1909.10140.pdf
    A NEW COEFFICIENT OF CORRELATION
    :return:
    """
    has_ties = has_duplicates(x)
    n = len(x)
    y_2 = sort_array_by_reference(y, x)
    ranks = ranks_of_array(y_2)
    if has_ties:
        inverse_ranks = ranks_of_array(-y_2)
        return 1 - n / 2 * (np.sum(np.abs(ranks[:-1] - ranks[1:]))) / (np.sum(
            inverse_ranks * (n - inverse_ranks)))
    else:
        return 1 - (3 / (n * n - 1)) * (np.sum(np.abs(ranks[:-1] - ranks[1:])))
