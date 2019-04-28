import random
import numpy as np
from utils import gaussian_kernel, nystorm_approximation
from numpy.linalg import inv


class Meka:

    def __init__(self, df, gamma, rank_k, number_clusters_c, number_sample_m):
        """
        Initialize the algorithm.
        :param df: Dataframe with the values
        :param gamma: float, parameter for the gaussian shift.
        :param rank_k: int, rank of the low-rank approximation matrix.
        :param number_clusters_c: int, number of blocks (clusters)
        :param number_sample_m: int, size of the sample for the decomposition.
        :return: G_hat, matrix based on the MEKA algorith
        """
        self.df = df
        self.gamma = gamma
        self.rank_k = rank_k
        self.number_clusters_c = number_clusters_c
        self.number_sample_m = number_sample_m

        self.blocks_of_L = {}
        self.blocks_of_W = []

    def compute_on_diagonal_blocks(self):
        """
        Compute the on-diagonal blocks by using the nystorm approximation for each block.
        :return:
        """
        for cluster in range(self.number_clusters_c):
            df_cluster = self.df.loc[self.df['clusters'] == cluster]  # Extract the cluster
            df_cluster = df_cluster.iloc[:, :-1]  # Delete the last column

            W, L = nystorm_approximation(df_cluster, self.number_sample_m, self.rank_k, self.gamma)

            # We can get the block G(s,s) by doing:
            # G = W * M_k * transpose(W)

            self.blocks_of_W.append(W)
            position = str(cluster) + '_' + str(cluster)
            self.blocks_of_L[position] = L  # Store the L(s,s) element

            print("Success for the cluster " + str(cluster))

    def compute_off_diagonal_blocks(self):
        """
        Compute the off-diagonal blocks. We use the blocks of W calculated previously.
        :return:
        """

        dim_subsample = 3 * self.rank_k  # Get the size of the subsample matrix
        for s in range(self.number_clusters_c):
            for t in range(self.number_clusters_c):
                if s != t:

                    df_s = self.df.loc[self.df['clusters'] == s].iloc[:, :-1]
                    df_t = self.df.loc[self.df['clusters'] == t].iloc[:, :-1]

                    number_col_s = df_s.shape[0]
                    number_row_t = df_t.shape[0]

                    index_col_s = random.randint(0, number_col_s - dim_subsample)
                    index_row_t = random.randint(0, number_row_t - dim_subsample)

                    # We have the coordinates of the submatrix

                    sub_matrix = np.zeros((dim_subsample, dim_subsample))

                    for i in range(index_row_t, index_row_t + dim_subsample):
                        for j in range(index_col_s, index_col_s + dim_subsample):
                            sub_matrix[i - index_row_t, j - index_col_s] = gaussian_kernel(self.gamma, df_s.iloc[j],
                                                                                           df_t.iloc[i])

                    # Then we can extract W_nu_s and W_nu_t

                    W_s = self.blocks_of_W[s]
                    W_nu_s = W_s[index_col_s:index_col_s + dim_subsample, :]

                    W_t = self.blocks_of_W[t]
                    W_nu_t = W_t[index_row_t:index_row_t + dim_subsample, :]

                    # And finally we compute the dot product (i.e solve Least Squares) to get L(s,t)

                    L_s_t = inv(np.transpose(W_nu_s).dot(W_nu_s)).dot(np.transpose(W_nu_s)).dot(sub_matrix).dot(
                        W_nu_t).dot(
                        inv(np.transpose(W_nu_t).dot(W_nu_t)))

                    self.blocks_of_L[str(s) + '_' + str(t)] = L_s_t

                    # We have now all the blocks of L and W

        print("Computation of off-diagonal blocks done.")

    def reconstruct_G_hat(self):
        """
        Reconstruct G hat based on the blocks decomposition.
        :return: the matrix G_hat of size (n x n)
        """
        n = len(self.df)
        G_hat = np.zeros((n, n))

        index_row = 0
        for i in range(self.number_clusters_c):
            W_1 = self.blocks_of_W[i]
            index_col = 0
            for j in range(self.number_clusters_c):
                W_2 = np.transpose(self.blocks_of_W[j])
                L = self.blocks_of_L[str(i) + '_' + str(j)]
                G_partial = W_1.dot(L).dot(W_2)
                a, b = G_partial.shape
                G_hat[index_row:index_row + a, index_col:index_col + b] = G_partial

                index_col += b
            index_row += a

        print('Construction of G_hat is finished.')
        return G_hat