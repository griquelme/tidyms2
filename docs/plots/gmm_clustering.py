import numpy as np
import matplotlib.pyplot as plt

from tidyms2.annotation.correspondence import cluster_dbscan, split_cluster_gmm

np.random.seed(1234)
n = 200
X1 = np.random.normal(size=(n, 2))
samples = np.hstack((np.arange(n), np.arange(n)))
X2 = np.random.normal(size=(n, 2), loc=(2, 2))
X = np.vstack((X1, X2))

dbscan_labels = cluster_dbscan(X, 2.0, 50, 10000)
gmm_labels, score = split_cluster_gmm(X, samples, 2, 3.0)

fig, ax = plt.subplots()
for l in np.unique(gmm_labels):
    ax.scatter(*X[gmm_labels == l].T, label=l)

ax.set_xlabel("m/z")
ax.set_ylabel("Rt")
ax.legend(title="GMM labels")