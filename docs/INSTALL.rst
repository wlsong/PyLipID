
============
Installation
============

To install the PyLipID Python package, you need a few Python package dependencies. These dependencies
include:

- mdtraj
- numpy
- pandas
- matplotlib
- networkx
- scipy
- logomaker
- scikit-learn
- tqdm
- kneebow

If these dependencies are not available in their required versions, the installation will fail. We thus
recommend using the package installer pip:

.. code-block:: bash

  $ pip install pylipid

If you know what you are doing and want to build from the source, the package is also available for
download on Github via:

.. code-block:: bash

  $ git clone https://github.com/pstansfeld/PyLipID.git
  $ cd PyLipID
  $ pip install .
