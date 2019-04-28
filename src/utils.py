import numpy as np


def nystorm_approximation(df, number_sample_m, rank_k, gamma):
    """
    Compute the Nystorm approximation of the dataframe.

    :param df: The dataframe which has the data points.
    :param number_sample_m: The size of the sample to take.
    :param rank_k: The rank of the matrix for the approximation.
    :param gamma: float, the parameter for the shift
    :return: W (matrix of size n x m) and L (matrix of size m x m)
    """

    sample = df.sample(n=number_sample_m, random_state=0, axis=0)

    kernel_matrix = get_kernel_matrix(sample, gamma)  # Compute the kernel matrix
    M_k = low_rank_approx(kernel_matrix, rank_k)  # Get the pseudo-inverse of the low rank approximation
    L = M_k
    W = get_decomposition(sample, df, gamma)  # Get the vector of the decomposition

    return W, L


def get_decomposition(sample, df, gamma):
    """
    Compute the decomposition of the matrix W.
    :param sample: matrix of size m x d
    :param df: matrix of size n x d
    :param gamma: float, parameter for the gaussian shift
    :return: The W matrix of size n x m
    """
    number_sample_m = len(sample)
    W = np.zeros((len(df), number_sample_m))  # Create the matrix
    for j in range(number_sample_m):
        for i in range(len(df)):
            W[i, j] = gaussian_kernel(gamma, sample.iloc[j], df.iloc[i])

    return W


def gaussian_kernel(gamma, x_1, x_2):
    """
    Compute the gaussian kernel between two vectors.

    :param gamma: float, gamma value
    :param x_1: array
    :param x_2: array
    :returns: float, the kernel value
    """
    return np.exp((-1) * gamma * np.linalg.norm(x_1 - x_2)**2)


def low_rank_approx(matrix, rank):
    """
    Compute the pseudo-inverse of a low k-rank approximation of a matrix.
    :param matrix: matrix to get the approximation
    :param rank: int, the rank of the matrix approximation
    :returns: a matrix which is the pseudo-inverse of the approximation
    """

    u, s, v = np.linalg.svd(matrix, full_matrices=False)

    M_k = np.zeros((len(u), len(v)))
    for i in range(rank):
        M_k += s[i] * np.outer(u.T[i], v[i])
    pseudo_inverse_M_k = np.linalg.pinv(M_k)
    return pseudo_inverse_M_k


def get_kernel_matrix(dataframe, gamma):
    """
    Take dataframe and compute the kernel matrix associated.
    :param dataframe: a df which size is (n x d)
    :param gamma: float, parameter for the shift of the gaussian kernel
    :return: matrix, the kernel matrix (of size n x n)
    """
    n = len(dataframe)
    kernel_matrix = np.zeros((n, n))  # Initialize the kernel matrix
    for i in range(n):
        for j in range(0, i):
            kernel_matrix[i, j] = gaussian_kernel(gamma, dataframe.iloc[i], dataframe.iloc[j])

    # Then we do a transposition because the kernel matrix is symetric
    return kernel_matrix + kernel_matrix.transpose()
