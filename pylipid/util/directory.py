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


def check_dir(root, name=None, print_info=True):
    """
    Create (if needed) and return a directory under 'root'.
    If 'name' is provided, the directory will be 'root/name'.
    Set print_info=False to suppress the creation message.
    """
    import os

    path = os.path.join(root, name) if name is not None else root

    if not os.path.isdir(path):
        if print_info:
            print(f"Creating new directory: {path}")
        os.makedirs(path, exist_ok=True)

    return path

