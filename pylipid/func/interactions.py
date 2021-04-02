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

"""This module contains the class Durations for using dual-cutoff scheme.
"""
import numpy as np

__all__ = ["cal_contact_residues", "Duration", "cal_occupancy", "cal_lipidcount"]


def cal_contact_residues(dist_matrix, cutoff):
    """Obtain contact residues as a function of time.

    Parameters
    ----------
    dist_matrix : list of lists or ndarray, shape=(n_residues, n_frames)
        The residue distances to the target. The distances of a residue
        to the target is listed in a row/a list as a function of time.
    cutoff : scalar
        The distance cutoff to define a contact. A distance to the target
        equal or lower to the `cutoff` is considered as in contact.

    Returns
    -------
    contact_list : list of lists
        A list of n_frame lists that contains the residues (represented by the row index of dist_matrix)
        within the cutoff distance to the target in each frame.
    frame_id_set : ndarray
        An array of frame indices where distances are smaller than the `cutoff`.
    residue_id_set : ndarray
        An array of residue indices which meet the distance

    Examples
    --------
    >>> dr0 = [0.9, 0.95, 1.2, 1.1, 1.0, 0.9] # the distances of R0 to the target as a function of time
    >>> dr1 = [0.95, 0.9, 0.95, 1.1, 1.2, 1.1] # the distances of R1
    >>> dr2 = [0.90, 0.90, 0.85, 0.95, 1.0, 1.1] # the distances of R2
    >>> dist_matrix = [dr0, dr1, dr2]
    >>> contact_list, frame_id_set, residue_id_set = cal_contact_residues(dist_matrix, 1.0)
    >>> print(contact_list)
    [[0, 1, 2], [0, 1, 2], [1, 2], [2], [0, 2], [0]]
    >>> print(frame_id_set)
    array([0, 1, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4])
    >>> print(residue_id_set)
    array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])

    """
    residue_id_set, frame_id_set = np.where(np.array(dist_matrix) <= cutoff)
    contact_list = [[] for dummy in np.arange(len(dist_matrix[0]))]
    for frame_id, residue_id in zip(frame_id_set, residue_id_set):
        contact_list[frame_id].append(residue_id)
    return contact_list, frame_id_set, residue_id_set


class Duration:
    def __init__(self, contact_low, contact_high, dt):
        """Dual cutoff scheme for calculating the interaction durations.

        Parameters
        ----------
        contact_low : list of lists
            A list of n_frame lists that contains the residues within the smaller
            distance as a function of trajectory frames.
        contact_high : list of lists
            A list of n_frame lists that contains the residues within the larger
            distance as a function of trajectory frames.
        dt : scalar
            The timestep between two adjacent trajectory frames.

        """
        self.contact_low = contact_low
        self.contact_high = contact_high
        self.dt = dt
        self.pointer = [np.zeros_like(self.contact_high[idx], dtype=np.int)
                        for idx in range(len(self.contact_high))]
        return

    def cal_durations(self):
        """Calculate interaction durations using the dual-cutoff scheme.

        Calculate the durations of the appearances that start from the point when an object appears
        in `contact_low` to the point when the object disappears from `contact_high`.

        Returns
        -------
        durations : list
            A list of durations of all the interactions defined by *contact_low* and *contact_high*

        """

        durations = []
        for i in range(len(self.contact_low)):
            for j in range(len(self.contact_low[i])):
                pos = np.where(self.contact_high[i] == self.contact_low[i][j])[0][0]
                if self.pointer[i][pos] == 0:
                    durations.append(self._get_duration(i, pos))
        if len(durations) == 0:
            return [0]
        else:
            durations.sort()
            return durations

    def _get_duration(self, i, j):
        count = 1
        self.pointer[i][j] = 1
        lipid_to_search = self.contact_high[i][j]
        for k in range(i+1, len(self.contact_high)):
            locations = np.where(self.contact_high[k] == lipid_to_search)[0]
            if len(locations) == 0:
                return count * self.dt
            else:
                pos = locations[0]
                self.pointer[k][pos] = 1
                count +=1
        return (count - 1) * self.dt


def cal_occupancy(contact_list):
    """Calculate the percentage of frames in which a contact is formed.

    Parameters
    ___________
    contact_list : list of lists
        A list of lists that contains the residues (represented by the row index of dist_matrix)
        within the cutoff distance to the target in each frame.

    Returns
    -------
    Ocupancy : scalar
        The percentage of frames in which a contact is formed

    See also
    --------
    pylipid.func.cal_contact_residues
        The function that calculates contact residues from distance matrix.
    pylipid.func.cal_lipidcount
        The function that calculates the average number of contacts in all frames

    """
    if len(contact_list) == 0:
        return 0, 0
    else:
        contact_counts = [len(item) for item in contact_list]
        mask = np.array(contact_counts) > 0
        contact_counts_nonzero = np.array(contact_counts)[mask]
        return 100 * len(contact_counts_nonzero)/len(contact_list)


def cal_lipidcount(contact_list):
    """Calculate the average number of contacts at a frame.

    Parameters
    ___________
    contact_list : list of lists
        A list of lists that contains the residues (represented by the row index of dist_matrix)
        within the cutoff distance to the target in each frame.

    Returns
    -------
    LipidCount : scalar
        The average number of contacts in a frame.

    See also
    --------
    pylipid.func.cal_contact_residues
        The function that calculates contact residues from distance matrix.
    pylipid.func.cal_occupancy
        The function that calculates the percentage of frames in which a contact is formed

    """
    if len(contact_list) == 0:
        return 0, 0
    else:
        contact_counts = [len(item) for item in contact_list]
        mask = np.array(contact_counts) > 0
        contact_counts_nonzero = np.array(contact_counts)[mask]

        if len(contact_counts_nonzero) == 0:
            return 0
        else:
            return np.nan_to_num(contact_counts_nonzero.mean())

