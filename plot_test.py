import pickle
import numpy as np
import matplotlib.pyplot as plt
import os.path

from test import ALGORITHMS


algorithms = list(ALGORITHMS) + ["SAC_model.zip"]
version="v_1"
figure_directory = os.path.join("results", f"resuts_{version}")

os.makedirs(figure_directory, exist_ok=True)

for algorithm in algorithms:
    print(f"Working on algorithm: {algorithm}")
    file_path = f"./pickle_files/pickle_files_{version}/test_data_{algorithm}.pickle"
    with open(file_path, 'rb') as f:
        test_data = pickle.load(f)
    algorithm_data = test_data[algorithm]

    user_in_high_rel_zone_list = []
    user_outage_list = []
    user_total_power_list =[]
    handover_rate_list =[]

    for sublist in algorithm_data:
        for dictionary in sublist:
            user_in_high_rel_zone_list.append(dictionary["user_in_high_rel_zone"])
            user_outage_list.append(dictionary["outage_probabilities"])
            user_total_power_list.append(dictionary["total_power_per_user"])
            handover_rate_list.append(dictionary["handover_rate"])

    user_in_high_rel_zone_array = np.concatenate(user_in_high_rel_zone_list)
    user_outage_array = np.concatenate(user_outage_list)
    user_total_power_array = np.concatenate(user_total_power_list)  
    #total_power_per_user
    user_total_power_in_the_box  = user_total_power_array[user_in_high_rel_zone_array]
    user_total_power_outside_box = user_total_power_array[~user_in_high_rel_zone_array]
    #outage_per_user
    outage_in_the_box = user_outage_array[user_in_high_rel_zone_array]
    outage_outside_box = user_outage_array[~user_in_high_rel_zone_array]

   


    # Plot and save the outage figure
    fig, ax = plt.subplots()
    ax.hist(np.log10(outage_outside_box), bins=10000, density=True, cumulative=True, histtype='step', label='Outside High Rel. Zone')
    ax.hist(np.log10(outage_in_the_box), bins=10000, density=True, cumulative=True, histtype='step', label='Inside High Rel. Zone')
    ax.legend()
    ax.set_xlabel('Log10 Outage')
    ax.set_ylabel('CDF')
    ax.set_title(algorithm)

    # # Save the figure in the specified directory
    # figure_path = os.path.join(figure_directory, f"{algorithm}_outage_figure.png")
    # plt.savefig(figure_path)
    # plt.close(fig)



    # Plot and save the total power figure
    fig, ax = plt.subplots()
    ax.hist(user_total_power_outside_box, bins=10000, density=True, cumulative=True, histtype='step', label='Outside High Rel. Zone')
    ax.hist(user_total_power_in_the_box, bins=10000, density=True, cumulative=True, histtype='step', label='Inside High Rel. Zone')
    ax.legend()
    ax.set_xlabel('Cluster power per user')
    ax.set_ylabel('CDF')
    ax.set_title(algorithm)

    # # Save the figure in the specified directory
    # figure_path = os.path.join(figure_directory, f"{algorithm}_power_figure.png")
    # plt.savefig(figure_path)
    # plt.close(fig)
