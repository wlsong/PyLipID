##############################################################################
# PyLipID: A python module for analysing protein-lipid interactions
#
# Author: Wanling Song
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
##############################################################################

"""This module contains functions for clustering the bound poses. """

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneebow.rotor import Rotor


__all__ = ["cluster_DBSCAN", "cluster_KMeans"]


def cluster_DBSCAN(data, eps=None, min_samples=None, metric="euclidean"):
    """Cluster data using DBSCAN.

    The the density-based spatial cluster `sklearn.cluster.DBSCAN
    <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_
    to cluster the data. If not provided by users, the distance cutoff `eps` is determined
    by the 'Knee method' which finds the distance at which a sharp change happens.

    Parameters
    ----------
    data : ndarray, shape=(n_samples, n_dims)
    eps : None or scalar, default=None
    min_samples : None or scalar, default=None
    metric : string or callable, default=’euclidean’
        The metric to use when calculating distance between instances in a feature array. If metric
        is a string or callable, it must be one of the options allowed by `sklearn.metrics.pairwise_distances`
        for its metric parameter.
    plot_dist_curve : bool, default=True

    Returns
    -------
    labels : array_like, shape=(n_samples)

    """
    if len(data) <= 3:
        return np.array([0 for dummy in data])
    if eps is None:
        nearest_neighbors = NearestNeighbors(n_neighbors=3)
        nearest_neighbors.fit(data)
        distances, indices = nearest_neighbors.kneighbors(data)
        distances = np.sort(distances, axis=0)[:, 1]
        data_vstacked = np.vstack([np.arange(len(distances)), distances]).T
        rotor = Rotor()
        rotor.fit_rotate(data_vstacked)
        elbow_index = rotor.get_elbow_index()
        eps = distances[elbow_index]
    if min_samples is None:
        scores = []
        for n_sample in np.arange(2, len(data)-1, 2):
            dbscan = DBSCAN(eps=eps, min_samples=n_sample, metric=metric)
            dbscan.fit(data)
            labels = dbscan.labels_
            if np.all(labels == -1):
                break
            else:
                scores.append(silhouette_score(data, labels))
        min_samples = np.arange(2, len(data)-1, 2)[np.argmax(scores)] # the highest silhouette_score.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan.fit(data)
    return dbscan.labels_


def cluster_KMeans(data, n_clusters):
    """Cluster data using KMeans.

    Parameters
    ----------
    data : ndarray, shape=(n_samples, n_dims)
    n_clusters : scalar

    Returns
    -----------
    labels : array_like, shape=(n_samples)

    """
    if len(data) < n_clusters:
        return cluster_DBSCAN(data, eps=None, min_samples=None, metric="euclidean")
    model = KMeans(n_clusters=n_clusters).fit(data)
    labels = model.predict(data)
    return labels