import kmeans1d
import numpy as np
import pandas as pd
from gap_statistic import OptimalK
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def kmeans(x,k):

    clusters, centroids = kmeans1d.cluster(x, k)
    return clusters,centroids

def gap(x):
    optimalK = OptimalK()
    n_clusters = optimalK(x, cluster_array=np.arange(1, 5))
    return n_clusters

if __name__ == '__main__':
    
    
    optimalK = OptimalK()

    x, y = make_blobs(n_samples=int(1e3), n_features=1, centers=3, random_state=25)
    print('Data shape: ', x.shape)

    n_clusters = optimalK(x, cluster_array=np.arange(1, 5))
    print('Optimal clusters: ', n_clusters)

    #print(optimalK.gap_df.head())
    
    #x = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
    k = n_clusters
    clusters,centroids = kmeans(x,k)
    print(clusters)   # [1, 1, 1, 0, 3, 3, 3, 2, 2, 2]
    print(centroids)  # [-50.0, 4.1, 94.0, 200.5]
    