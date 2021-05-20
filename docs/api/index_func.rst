
pylipid.func
==============

The ``func`` package provides tools to collect lipid interactions, calculate koff/residence time,
and calculate binding sites in PyLipID


.. currentmodule:: pylipid.func

.. rubric:: binding_site

.. autosummary::
   :toctree: generated/

   ~get_node_list
   ~collect_bound_poses
   ~vectorize_poses
   ~calculate_scores
   ~write_bound_poses


.. rubric:: clusterer

.. autosummary::
   :toctree: generated/

    ~cluster_DBSCAN
    ~cluster_KMeans


.. rubric:: interactions

.. autosummary::
   :toctree: generated/

    ~cal_contact_residues
    ~Duration
    ~cal_occupancy
    ~cal_lipidcount


.. rubric:: kinetics

.. autosummary::
   :toctree: generated/

    ~cal_koff
    ~cal_survival_func
