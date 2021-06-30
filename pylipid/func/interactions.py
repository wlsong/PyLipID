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

    This function takes a distance matrix that records the measured distance for molecules at each trajectory frame,
    then returns the indices of molecules the distance of which are smaller than the provided cutoff at each frame. It
    also returns the molecule indices the frame indices in which the molecule is within the cutoff.

    Parameters
    ----------
    dist_matrix : list or numpy.ndarray, shape=(n_residues, n_frames)
        The measured distance for molecules at each trajectory frame.

    cutoff : scalar
        The distance cutoff to define a contact. A distance to the target
        equal or lower to the ``cutoff`` is considered as in contact.

    Returns
    -------
    contact_list : list
        A list that records the indices of molecules that are within the given cutoff in each frame

    frame_id_set : list
        A list of frame indices for contacting molecules.

    residue_id_set : lsit
        A list of contacting molecules indices.

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

        In the dual cutoff scheme, a continuous contact starts when a molecule moves closer than the lower distance cutoff
        and ends when the molecule moves out of the upper cutoff. The duration between these two time points is the
        duration of the contact.

        Here, the ``contact_low`` is the lipid index for the lower cutoff and ``contact_high`` is the lipid index
        for the upper cutoff. For calculation of contact durations, a lipid molecule that appears in the ``contact_low``
        is searched in the subsequent frames of the ``contact_high`` and the search then stops if this
        molecule disappears from the ``contact_high``. This lipid molecule is labeled as 'checked', and the duration of
        this contact is calculated from the number of frames in which this lipid molecule appears in the lipid indices.
        This calculation iterates until all lipid molecules in the lower lipid index are labeled as 'checked'.

        Parameters
        ----------
        contact_low : list
            A list that records the indices of lipid molecule within the lower distance cutoff at each trajectory frame.

        contact_high : list
            A list that records the indices of lipid molecule within the upper distance cutoff at each trajectory frame.

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

        Calculate the durations of the appearances that start from the point when a molecule appears
        in ``contact_low`` and ends when it disappears from ``contact_high``.

        Returns
        -------
        durations : list
            A list of durations of the contacts defined by ``contact_low`` and ``contact_high``.

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

    ``contact_list`` records a list of residue indices of contact lipid molecules at each trajectory frames. This function
    calculates the percentage of frames that a lipid contact is formed.

    Parameters
    ___________
    contact_list : list
        A list of residue indices of contact lipid molecules at each trajectory frames.

    Returns
    -------
    Ocupancy : scalar
        The percentage of frames in which a contact is formed

    Examples
    --------
    >>> contact_list = [[], [130], [130, 145], [145], [], [], [145], [145]] # contacts are formed in 5 out of the 8 frames
    >>> occupancy = cal_occupancy(contact_list)
    >>> print(occupancy) # percentage
    62.5

    See also
    --------
    pylipid.api.LipidInteraction.compute_residue_occupancy
        Calculate the percentage of frames in which the specified residue formed lipid contacts for residues.
    pylipid.api.LipidInteraction.compute_site_occupancy
        Calculate the percentage of frames in which the specified lipid contacts are formed for binding sites.

    """
    if len(contact_list) == 0:
        return 0
    else:
        contact_counts = [len(item) for item in contact_list]
        mask = np.array(contact_counts) > 0
        return 100 * np.sum(mask)/len(contact_list)


def cal_lipidcount(contact_list):
    """Calculate the average number of contacting molecules.

    This function calculates the average number of contacting molecules when any contact is formed.

    Parameters
    ___________
    contact_list : list
        A list of residue indices of contact lipid molecules at each trajectory frames.

    Returns
    -------
    LipidCount : scalar
        The average number of contacts in frames in which any contact is formed.

    Examples
    --------
    >>> contact_list = [[], [130], [130, 145], [145], [], [], [145], [145]]
    >>> lipidcount = cal_lipidcount(contact_list)
    >>> print(lipidcount) # (1+2+1+1+1)/5
    1.2

    See also
    --------
    pylipid.api.LipidInteraction.compute_residue_lipidcount
        Calculate the average number of contacting lipids for residues.
    pylipid.api.LipidInteraction.compute_site_lipidcount
        Calculate the average number of contacting lipids for binding sites.

    """
    if len(contact_list) == 0:
        return 0
    else:
        contact_counts = np.array([len(item) for item in contact_list])
        mask = contact_counts > 0
        if np.sum(mask) == 0:
            return 0
        else:
            contact_counts_nonzero = contact_counts[mask]
            return np.nan_to_num(contact_counts_nonzero.mean())

