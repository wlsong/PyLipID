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
    labels : array_like, shape=(n_samples,)
    core_sample_indices : array_like, shape=(n_clusters,)

    """
    if len(data) <= len(data[0]):
        return np.array([0 for dummy in data]), np.arange(len(data))[np.newaxis, :]
    if len(data) < 200:
        min_samples = 2 * len(data[0])
    elif 200 <= len(data) < 800:
        min_samples = 5 * len(data[0])
    elif len(data) >= 800:
        min_samples = 10 * len(data[0])
    if eps is None:
        nearest_neighbors = NearestNeighbors(n_neighbors=min_samples)
        nearest_neighbors.fit(data)
        distances, indices = nearest_neighbors.kneighbors(data)
        distances = np.sort(distances, axis=0)[:, 1]
        data_vstacked = np.vstack([np.arange(len(distances)), distances]).T
        rotor = Rotor()
        rotor.fit_rotate(data_vstacked)
        elbow_index = rotor.get_elbow_index()
        eps = distances[elbow_index]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan.fit(data)
    core_sample_indices = [[] for label in np.unique(dbscan.labels_) if label != -1]
    for core_sample_index in dbscan.core_sample_indices_:
        core_sample_indices[dbscan.labels_[core_sample_index]].append(core_sample_index)
    return dbscan.labels_, core_sample_indices


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