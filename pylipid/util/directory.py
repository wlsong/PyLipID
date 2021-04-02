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

"""This module contains the assisting functions for dealing with directories."""

import os

__all__ = ["check_dir"]


def check_dir(directory=None, suffix=None, print_info=True):
    """Creating directory

    This function will combine the suffix with the given directory (or the current
    working directory if none is given) to generate a directory name, and create a
    directory with this name if it does not exit.

    """
    if directory is None:
        directory = os.getcwd()
    else:
        directory = os.path.abspath(directory)
    if suffix is not None:
        directory = os.path.join(directory, suffix)
    if not os.path.isdir(directory):
        os.makedirs(directory)
        if print_info:
            print("Creating new director: {}".format(directory))
    return directory

