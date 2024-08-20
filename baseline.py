import numpy as np

def power_alo_for_closest_bs(observation: dict, info: dict, state=None):
    bs_locs = info.get("bs_locations")
    user_locs = observation['user_locations']
    status = observation['status']
    user_locs = np.array(user_locs)
    bs_locs = np.array(bs_locs)
    dists = np.sqrt(np.sum((user_locs[:, np.newaxis] - bs_locs) ** 2, axis=2))
    closest_bs_indices = np.argmin(dists, axis=1)
    bs_mask = np.where(np.arange(len(bs_locs)) == closest_bs_indices[:, np.newaxis], 1, 0)
    # Return power
    power_allocation = bs_mask * (status[:, np.newaxis] == 1)
    return power_allocation, state

def power_alo_from_all_bs(observation: dict, info: dict, state=None):
    n_users = len(observation["user_locations"])
    n_base_stations = len(info["bs_locations"])
    status = observation['status']
    matrix = np.zeros((n_users, n_base_stations))
    matrix[status == 1] = 1
    return matrix, state

def fixed_clustering(observation: dict, info: dict, state=None,
                     num_user_per_bs: int=2):
    status = observation['status']
    bs_locs = info.get("bs_locations")
    user_locs = observation['user_locations']
    n_users = len(user_locs)
    n_bs = len(bs_locs)
    if state is None: # or change in active users
        last_status = np.zeros((n_users, n_bs))
    else:
        last_status, clustering = state

    if state is None or np.any(status-last_status):
        _user_locs = np.expand_dims(user_locs, 1)
        _bs_locs = np.tile(bs_locs, (n_users, 1, 1))
        idx_active_users = np.where(status)[0]
        num_active_users = len(idx_active_users)
        _distances = np.linalg.norm(_user_locs - _bs_locs, ord=2, axis=2)
        num_user_per_bs = np.minimum(num_user_per_bs, num_active_users)
        k_closest_bs = np.argpartition(_distances[idx_active_users], num_user_per_bs, axis=0)
        idx_clusters = k_closest_bs[:num_user_per_bs]
        _assignment = np.zeros((num_active_users, n_bs))
        _assignment[idx_clusters, np.arange(n_bs)] = 1.
        clustering = np.zeros((n_users, n_bs))
        clustering[idx_active_users] = _assignment
        #clustering[idx_active_users][idx_clusters, np.arange(n_bs)] = 1.
    state = (status, clustering)
    power_allocation = clustering
    return power_allocation, state

if __name__ == "__main__":

    uav_locations = [
        [10, 10, 50],
        [50, 50, 10],
        [80, 80, 30],
        [250, 125, 15],
        [327, 234, 60],
    ]
    bs_locations = [[100, 100, 50], [100, 400, 25], [400, 400, 15], [400, 100, 30]]
    max_power_db =30
    bs_powers = power_alo_for_closest_bs(uav_locations,bs_locations)
    print(bs_powers)
