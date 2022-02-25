import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans # Used in k-means Clustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# IMPORT DATASET
def data(dataset, vector):
    dataset = pd.read_csv(dataset)
    X = dataset.iloc[:, vector].values

    return X


# USE ELBOW METHOD TO FIND OPTIMAL NUMBER OF CLUSTERS
def elbow_method(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    return wcss

def dendrogram_method(X):
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()
    return dendrogram


# TRAIN K-MEANS MODEL ON DATASET
def k_means(X, nClusters = 5, init_ = 'k-means++', randomState = 42)
    kmeans = KMeans(n_clusters = nClusters, init = init_, random_state = randomState)
    y_kmeans = kmeans.fit_predict(X)
    return kmeans, y_kmeans

# TRAIN HIERARCHICAL CLUSTERING MODEL ON DATASET
def hierarchical(X, nClusters = 5, affinity_ = 'euclidean', linkage_ = 'ward'):
    hc = AgglomerativeClustering(n_clusters = nClusters, affinity = affinity_, linkage = linkage_)
    y_hc = hc.fit_predict(X)
    return hc, y_hc

# VISUALIZE CLUSTERS
def visualization(X, y_kmeans, kmeans, title, xlabel, ylabel)
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
