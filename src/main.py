from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from utils import get_kernel_matrix, nystorm_approximation
from meka import Meka

df = pd.read_csv('data/ijcnn1.csv', sep=',').iloc[:, 1:]
df = df.sample(n=500, random_state=0)
rank_k = 10
number_sample_m = 300

gamma = 1

number_clusters_c = 3
# Do the Kmeans clustering and group data points based on that
kmeans = KMeans(n_clusters=number_clusters_c, random_state=0).fit(df)

df['clusters'] = kmeans.labels_
df_c = df.sort_values(by='clusters')
df = df.drop(['clusters'], axis=1)

meka_algorithm = Meka(df_c, gamma, rank_k, number_clusters_c, number_sample_m=20)
meka_algorithm.compute_on_diagonal_blocks()
meka_algorithm.compute_off_diagonal_blocks()

G_hat = meka_algorithm.reconstruct_G_hat()


G = get_kernel_matrix(df, gamma=gamma)


W, L = nystorm_approximation(df, number_sample_m, rank_k, gamma)
G_nystrom = W.dot(L).dot(np.transpose(W))

score = np.linalg.norm(G - G_nystrom, 'fro') / np.linalg.norm(G, 'fro')

score_2 = np.linalg.norm(G - G_hat, 'fro') / np.linalg.norm(G, 'fro')


