# Creation of the Memory Kernel Approximation

from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.datasets import load_svmlight_file
import random


def nystorm_approximation(matrix, cluster):
    """
    Compute the Nystorm approximation.
    :param matrix: The dataframe which has the data points.
    :param cluster: The id of the current cluster
    :return:
    """

    sample = matrix.sample(n=number_sample_m, random_state=0, axis=0)

    kernel_matrix = get_kernel_matrix(sample)  # Compute the kernel matrix
    M_k = low_rank_approx(kernel_matrix, rank_k)  # Get the pseudo-inverse of the low rank approximation

    W = get_decomposition(sample, df_cluster)  # Get the vector of the decomposition

    # We can get the block G(s,s) by doing:
    # G = W * M_k * transpose(W)

    blocks_of_W.append(W)
    position = str(cluster) + '_' + str(cluster)
    blocks_of_L[position] = M_k  # Store the L(s,s) element

    print("Sucess for the cluster " + str(cluster))


def get_kernel_matrix(vectors):
    """
    Take vectors and compute the kernel matrix associated.
    :param vectors:
    :return: kernel matrix
    """
    number_m = len(vectors)
    kernel_matrix = np.zeros((number_m, number_m))  # Initialize the kernel matrix
    for i in range(int(number_m/2) - 1):
        for j in range(i+1, int(number_m / 2)):
            kernel_matrix[i, j] = gaussian_kernel(gamma, vectors.iloc[i], vectors.iloc[j])

    # Then we do a transposition because the kernel matrix is symetric
    return kernel_matrix + kernel_matrix.transpose()


def low_rank_approx(matrix, rank):
    """
    Compute the pseudo-inverse of a low k-rank approximation of a matrix.
    Requires: numpy
    """

    u, s, v = np.linalg.svd(matrix, full_matrices=False)

    M_k = np.zeros((len(u), len(v)))
    for i in range(rank):
        M_k += s[i] * np.outer(u.T[i], v[i])
    pseudo_inverse_M_k = np.linalg.pinv(M_k)
    return pseudo_inverse_M_k


def get_decomposition(vectors, matrix):
    """
    Compute the decomposition of the matrix C.
    :param vectors:
    :param matrix:
    :return:
    """
    number_sample_m = len(vectors)
    C = np.zeros((len(df_cluster), number_sample_m))  # Create the matrix
    for j in range(number_sample_m):
        for i in range(len(df_cluster)):
            C[i, j] = gaussian_kernel(gamma, vectors.iloc[j], matrix.iloc[i])

    return C


def gaussian_kernel(gamma, x_1, x_2):
    """
    Compute the gaussian kernel between two vectors.
    """
    return np.exp((-1) * gamma * np.linalg.norm(x_1 - x_2)**2)


rank_k = 3  # Rank
gamma = 0.001
number_clusters_c = 3  # Number of clusters
number_sample_m = 20

wine_dataset = load_wine(return_X_y=True)
X_wine = wine_dataset[0]

df = pd.DataFrame(X_wine)
n = len(df)

# Do the Kmeans clustering and group data points based on that
kmeans = KMeans(n_clusters=number_clusters_c, random_state=0).fit(X_wine)

df['clusters'] = kmeans.labels_
df = df.sort_values(by='clusters')

blocks_of_W = []
blocks_of_L = {}


for k in range(number_clusters_c):
    df_cluster = df.loc[df['clusters'] == k]  # Extract the cluster
    df_cluster = df_cluster.iloc[:, :-1]  # Delete the last column
    nystorm_approximation(df_cluster, k)

# Approximation done

dim_subsample = 3 * rank_k  # Get the size of the subsample matrix


def compute_off_diagonal_blocks(dim_subsample):
    for s in range(number_clusters_c):
        for t in range(number_clusters_c):
            if s != t:

                df_s = df.loc[df['clusters'] == s]
                df_t = df.loc[df['clusters'] == t]

                number_col_s = df_s.shape[0]
                number_row_t = df_t.shape[0]

                index_col_s = random.randint(0, number_col_s - dim_subsample)
                index_row_t = random.randint(0, number_row_t - dim_subsample)

                # We have the coordinates of the submatrix

                sub_matrix = np.zeros((dim_subsample,  dim_subsample))

                for i in range(index_row_t, index_row_t + dim_subsample):
                    for j in range(index_col_s, index_col_s + dim_subsample):
                        sub_matrix[i - index_row_t, j - index_col_s] = gaussian_kernel(gamma, df_s.iloc[j], df_t.iloc[i])

                # Then we can extract W_nu_s and W_nu_t

                W_s = blocks_of_W[s]
                W_nu_s = W_s[index_col_s:index_col_s + dim_subsample, :]

                W_t = blocks_of_W[t]
                W_nu_t = W_t[index_row_t:index_row_t + dim_subsample, :]

                # And finally we compute the dot product (i.e solve Least Squares) to get L(s,t)

                L_s_t = inv(np.transpose(W_nu_s).dot(W_nu_s)).dot(np.transpose(W_nu_s)).dot(sub_matrix).dot(W_nu_t).dot(
                    inv(np.transpose(W_nu_t).dot(W_nu_t)))

                blocks_of_L[str(s) + '_' + str(t)] = L_s_t


# We have now all the blocks of L and W

# We can construct G_hat:


def reconstruct_G_hat():
    """
    Reconstruct G hat based on the blocks decomposition.
    :return:
    """
    G_hat = np.zeros((n, n))

    index_row = 0
    for i in range(number_clusters_c):
        W_1 = blocks_of_W[i]
        index_col = 0
        for j in range(number_clusters_c):
            W_2 = np.transpose(blocks_of_W[j])
            L = blocks_of_L[str(i) + '_' + str(j)]
            G_partial = W_1.dot(L).dot(W_2)
            a, b = G_partial.shape
            G_hat[index_row:index_row+a, index_col:index_col+b] = G_partial

            index_col += b
        index_row += a

    return G_hat





