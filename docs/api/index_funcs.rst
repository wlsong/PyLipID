
funcs (pylipid.funcs)
=====================

The ``funcs`` package provides tools to collect lipid interactions, calculate koff/residence time,
and calculate binding sites in PyLipID


.. rubric:: binding_site

.. autosummary::
   :toctree: generated/

   pylipid.funcs.get_node_list
   pylipid.funcs.collect_bound_poses
   pylipid.funcs.vectorize_poses
   pylipid.funcs.calculate_scores
   pylipid.funcs.write_bound_poses
   pylipid.funcs.calculate_site_surface_area


.. rubric:: clusterer

.. autosummary::
   :toctree: generated/

    pylipid.funcs.cluster_DBSCAN
    pylipid.funcs.cluster_KMeans


.. rubric:: interactions

.. autosummary::
   :toctree: generated/

    pylipid.funcs.cal_contact_residues
    pylipid.funcs.Duration
    pylipid.funcs.cal_interaction_frequency


.. rubric:: kinetics

.. autosummary::
   :toctree: generated/

    cal_koff
    cal_survival_func
