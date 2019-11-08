#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# PyLipID --- https://github.com/wlsong/PyLipID
# Copyright (c) 2019 The PyLipID Development Team and contributors
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of PyLipID in published work:
#

"""Setuptools-based setup script for PyLipID.
For a basic installation just type the command::
  python setup.py install
"""

from setuptools import setup, find_packages

RELEASE = "beta"

if __name__ == '__main__':
    LONG_DESCRIPTION = '''LONG_DESCRIPTION'''
    CLASSIFIERS = [
        'Development Status :: Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Operating System :: POSIX',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows ',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]

    install_requires = [
          'mdtraj',
          'networkx',
          'seaborn',
          'Pillow',
          "python_louvain",
          'pandas',
          'scipy',
          'matplotlib'
          'pymol',
    ]

    setup(name='PyLipID',
          version=RELEASE,
          description=('description'),
          long_description=LONG_DESCRIPTION,
          long_description_content_type='text/x-rst',
          author='Wanlin Song',
          author_email='wanling.song@bioch.ox.ac.uk',
          maintainer='Wanlin Song',
          maintainer_email='wanling.song@bioch.ox.ac.uk',
          url='https://github.com/wlsong/PyLipID',
          download_url='https://github.com/wlsong/PyLipID',
          project_urls={'Documentation': 'https://github.com/wlsong/PyLipID',
                        'Issue Tracker': 'https://github.com/wlsong/PyLipID/issues',
                        'Source': 'https://github.com/wlsong/PyLipID',
                        },
          license='GPL 2',
          classifiers=CLASSIFIERS,
          provides=['PyLipID'],
          packages=find_packages(),
          requires=['numpy', 'cython', 'mdtraj', 'networkx', 'seaborn', 'Pillow', 'community', 'pandas', 'scipy', 'matplotlib',],
          install_requires=install_requires,
          setup_requires=[
              'numpy',
              'cython',
          ],
          scripts=['pylipid.py', ],
          zip_safe=False,  # as a zipped egg the *.so files are not found (at
                           # least in Ubuntu/Linux)
    )
