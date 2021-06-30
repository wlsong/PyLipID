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
    r"""Cluster data using DBSCAN.

    This function clusters the samples using a density-based cluster
    `DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_ provided by scikit.
    DBSCAN finds clusters of core samples of high density. A sample point is a core sample if at least `min_samples`
    points are within distance :math:`\varepsilon` of it. A cluster is defined as a set of sample points that are
    mutually density-connected and density-reachable, i.e. there is a path
    :math:`\left\langle p_{1}, p_{2}, \ldots, p_{n}\right\rangle` where each :math:`p_{i+1}` is within distance
    :math:`\varepsilon` of :math:`p_{i}` for any two p in the two. The values of `min_samples` and :math:`\varepsilon`
    determine the performance of this cluster.

    If None, `min_samples` takes the value of 2 * n_dims. If :math:`\varepsilon` is None, it is set as the value at the
    knee of the k-distance plot.

    Parameters
    ----------
    data : numpy.ndarray, shape=(n_samples, n_dims)
        Sample data to find clusters.

    eps : None or scalar, default=None
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is
        not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to
        choose appropriately for your data set and distance function. If None, it is set as the value at the
        knee of the k-distance plot.

    min_samples : None or scalar, default=None
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This
        includes the point itself. If None, it takes the value of 2 * n_dims

    metric : string or callable, default=’euclidean’
        The metric to use when calculating distance between instances in a feature array. If metric
        is a string or callable, it must be one of the options allowed by `sklearn.metrics.pairwise_distances`
        for its metric parameter.

    Returns
    -------
    labels : array_like, shape=(n_samples,)
        Cluster labels for each data point.

    core_sample_indices : array_like, shape=(n_clusters,)
        Indices of core samples.

    """
    if len(data) <= len(data[0]):
        return np.array([0 for dummy in data]), np.arange(len(data))[np.newaxis, :]
    if 2*len(data[0]) > len(data):
        min_samples = np.min([len(data[0]), 4])
    elif len(data) < 1000:
        min_samples = np.min([2 * len(data[0]), len(data)])
    elif len(data) >= 1000:
        min_samples = np.min([5 * len(data[0]), len(data)])
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
    r"""Cluster data using KMeans.

    This function clusters the samples
    using `KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_
    provided by scikit. The KMeans cluster separates the samples into `n` clusters of equal variances, via minimizing
    the `inertia`, which is defined as:

    .. math::
        \sum_{i=0}^{n} \min _{u_{i} \in C}\left(\left\|x_{i}-u_{i}\right\|^{2}\right)

    where :math:`u_{i}` is the `centroid`  of cluster i. KMeans scales well with large dataset but performs poorly
    with clusters of varying sizes and density.

    Parameters
    ----------
    data : numpy.ndarray, shape=(n_samples, n_dims)
        Sample data to find clusters.

    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.

    Returns
    -----------
    labels : array_like, shape=(n_samples)
        Cluster labels for each data point.

    """
    if len(data) < n_clusters:
        return cluster_DBSCAN(data, eps=None, min_samples=None, metric="euclidean")
    model = KMeans(n_clusters=n_clusters).fit(data)
    labels = model.predict(data)
    return labels