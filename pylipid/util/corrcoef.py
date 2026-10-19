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

"""This module contains functions for calculating correlation coefficient. """

from scipy import sparse
import numpy as np

__all__ = ["sparse_corrcoef"]


def sparse_corrcoef(A, B=None):
    """Calculate correlation coeffient matrix using sparse matrix"""
    if B is not None:
        A = sparse.vstack((A, B), format='csr')
    A = A.astype(np.float64)
    n = A.shape[1]
    # Compute the covariance matrix.
    # Use explicit dense conversion to avoid numpy.matrix deprecation and
    # version-dependent sparse-dense arithmetic (scipy >= 1.14 / numpy >= 1.25).
    rowsum = np.asarray(A.sum(1)).flatten()          # shape (n_rows,) ndarray
    centering = np.outer(rowsum, rowsum) / n         # explicit outer product
    # Handle both sparse matrices (from collect_residue_contacts) and
    # plain ndarrays (from tests or direct calls).
    dot = A.dot(A.T.conjugate())
    if sparse.issparse(dot):
        dot = np.asarray(dot.todense())
    else:
        dot = np.asarray(dot)
    C = (dot - centering) / (n - 1)
    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    corrcoefs = C / np.sqrt(np.outer(d, d))
    return corrcoefs
