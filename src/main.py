from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.datasets import load_svmlight_file
import random
import MEKA
from sklearn.kernel_approximation import Nystroem

df = pd.read_csv('src/data/ijcnn1.csv', sep=',').iloc[:, 1:]
n = 500
df = df.sample(n=500, random_state=0)
rank = 10

gamma = 1
G = np.zeros((n, n))

for i in range(n):
    for j in range(0, i):
        G[i, j] = MEKA.gaussian_kernel(gamma, df.iloc[i], df.iloc[j])

# Then we do a transposition because the kernel matrix is symetric
G = G + G.transpose()


# Then the Nystrom approximation:

sample = df.sample(n=300, random_state=0, axis=0)

  # Compute the kernel matrix

n = len(sample)
kernel_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(0, i):
        kernel_matrix[i, j] = MEKA.gaussian_kernel(gamma, sample.iloc[i], sample.iloc[j])

kernel_matrix = kernel_matrix + np.transpose(kernel_matrix)


# Pseudo inverse of low rank approximation

u, s, v = np.linalg.svd(kernel_matrix, full_matrices=False)

M_k = np.zeros((len(u), len(v)))
for i in range(rank):
    M_k += s[i] * np.outer(u.T[i], v[i])
pseudo_inverse_M_k = np.linalg.pinv(M_k)


C = np.zeros((len(df), len(sample)))  # Create the matrix
for j in range(len(sample)):
    for i in range(len(df)):
        C[i, j] = MEKA.gaussian_kernel(gamma, sample.iloc[j], df.iloc[i])

W = C
G_nystrom = W.dot(pseudo_inverse_M_k).dot(np.transpose(W))







score = np.linalg.norm(G - G_nystrom, 'fro') / np.linalg.norm(G, 'fro')


