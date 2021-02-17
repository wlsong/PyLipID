
=====
Mics.
=====

Here we provide a no-brainer python script for lipid interaction analysis using PyLipID.::

    import numpy as np
    import matplotlib.pyplot as plt
    from pylipid.api import LipidInteraction

    ##################################################################
    ##### This part needs changes according to your setting ##########
    ##################################################################
    trajfile_list = ["run1/md.xtc", "run2/md.xtc"]
    topfile_list = ["run1/md.gro", "run2/md.gro"]
    lipid = "CHOL"
    cutoffs = [0.55, 0.8]
    nprot = 1

    #####################################
    ###### no changes needed below ######
    #####################################

    #### calculate lipid interactions
    li = LipidInteraction(trajfile_list, topfile_list, cutoffs=cutoffs, lipid=lipid, nprot=1)
    li.collect_residue_contacts(write_log=True, print_log=True)
    li.compute_residue_koff(print_data=False, plot_data=True, fig_close=True)
    li.compute_binding_nodes(threshold=4, print_data=False)
    li.compute_site_koff(print_data=True, plot_data=True, sort_residue="Residence Time", fig_close=True)
    _, pose_rmsd_data = li.analyze_bound_poses()
    surface_area_data = li.compute_surface_area()

    #### write and save data
    li.write_site_info()

    for item in ["Dataset", "Duration", "Occupancy", "Lipid Count", "CorrCoef",
                 "Duration BS", "Occupancy BS"]:
        li.save_data(item=item)

    for item in ["Dataset", "Duration", "Occupancy", "Lipid Count", "Duration BS",
                 "Occupancy BS", "Residence Time BS"]:
        li.save_coordinate(item=item)

    for item in ["Dataset", "Duration", "Occupancy", "Lipid Count"]:
        li.plot(item=item, fig_close=True)
        li.plot_logo(item=item, fig_close=True)

    pose_rmsd_data.to_csv("{}/pose_rmsd_data.csv".format(li.save_dir))
    surface_area_data.to_csv("{}/surface_area_data.csv".format(li.save_dir))


    #### plot binding site comparison.

    timeunit = timeunit = 'ns' if li.timeunit == "ns" else r"$\mu$s"
    ylabel_dict = {"Residence Time": "Residence Time ({})".format(timeunit),
                   "Duration": "Duration ({})".format(timeunit),
                   "Occupancy": "Occuoancy (100%)",
                   "Lipid Count": "Lipid Count (num.)"}
    binding_site_IDs = np.sort([int(bs_id) for bs_id in li.dataset["Binding Site ID"].unique() if bs_id != -1])

    # plot No. 1
    for item in ["Residence Time", "Duration", "Occupancy", "Lipid Count"]:
        item_values = np.array([li.dataset[li.dataset["Binding Site ID"]==bs_id]["Binding Site {}".format(item)].unique()[0]
                       for bs_id in binding_site_IDs])
        fig, ax = plt.subplots(1, 1)
        ax.scatter(np.arange(len(item_values)), np.sort(item_values)[::-1], s=50, color="red")
        ax.set_xticks(np.arange(len(item_values)))
        sorted_index = np.argsort(item_values)[::-1]
        ax.set_xticklabels(binding_site_IDs[sorted_index])
        ax.set_xlabel("Binding Site ID", fontsize=12)
        ax.set_ylabel(ylabel_dict[item], fontsize=12)
        for label in ax.xaxis.get_ticklabels()+ax.yaxis.get_ticklabels():
            plt.setp(label, fontsize=12, weight="normal")
        plt.tight_layout()
        plt.savefig("{}/{}_{}_v_binding_site.pdf".format(li.save_dir, li.lipid, "_".join(item.split())), dpi=200)
        plt.close()


    # plot No. 2
    RMSD_averages = np.array([pose_rmsd_data["Binding Site {}".format(bs_id)].dropna(inplace=False).mean()
                     for bs_id in binding_site_IDs])
    fig, ax = plt.subplots(1, 1)
    ax.scatter(np.arange(len(RMSD_averages)), np.sort(RMSD_averages)[::-1], s=50, color="red")
    ax.set_xticks(np.arange(len(RMSD_averages)))
    sorted_index = np.argsort(RMSD_averages)[::-1]
    ax.set_xticklabels(binding_site_IDs[sorted_index])
    ax.set_xlabel("Binding Site ID", fontsize=12)
    ax.set_ylabel("RMSD (nm)", fontsize=12)
    for label in ax.xaxis.get_ticklabels()+ax.yaxis.get_ticklabels():
        plt.setp(label, fontsize=12, weight="normal")
    plt.tight_layout()
    plt.savefig("{}/{}_RMSD_v_binding_site.pdf".format(li.save_dir, li.lipid), dpi=200)
    plt.close()


    # plot 3
    surface_area_averages = np.array([surface_area_data["Binding Site {}".format(bs_id)].dropna(inplace=False).mean()
                                      for bs_id in binding_site_IDs])
    fig, ax = plt.subplots(1, 1)
    ax.scatter(np.arange(len(surface_area_averages)), np.sort(surface_area_averages)[::-1], s=50, color="red")
    ax.set_xticks(np.arange(len(surface_area_averages)))
    sorted_index = np.argsort(surface_area_averages)[::-1]
    ax.set_xticklabels(binding_site_IDs[sorted_index])
    ax.set_xlabel("Binding Site ID", fontsize=12)
    ax.set_ylabel(r"Surface Area (nm$^2$)", fontsize=12)
    for label in ax.xaxis.get_ticklabels()+ax.yaxis.get_ticklabels():
        plt.setp(label, fontsize=12, weight="normal")
    plt.tight_layout()
    plt.savefig("{}/{}_surface_area_v_binding_site.pdf".format(li.save_dir, li.lipid), dpi=200)
    plt.close()
