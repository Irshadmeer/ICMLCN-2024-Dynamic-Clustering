from typing import Iterable

import numpy as np
from scipy import stats
from scipy import special
from scipy import linalg

from beamforming import calculate_phi, calculate_SyN, calculate_SzM, calculate_theta


def calculate_outage_probabilities(
    user_locs: np.array,
    bs_locs: np.array,
    bs_powers: np.array,
    carrier_freq: float,
    sinr_threshold: float,
    status: Iterable,
    los: np.array = None,
    num_antennas: int=16,
    bandwidth: float = 10e6,
    noise_den_db: float = -174,
):
    """Calculate the outage probabilities for each user

    Calculate the outage probabilities for each user given their positions and
    the power levels of the base stations.


    Parameters
    ----------
    user_locs : np.array
        Array of size :math:`N \\times 3`, where :math:`N` is the number of
        users/UAVs and the columns represent their :math:`(x, y, z)`
        coordinates.

    bs_locs : np.array
        Array of size :math:`B \\times 3`, where :math:`B` is the number of
        base stations and the columns represent their :math:`(x, y, z)`
        coordinates.

    bs_powers : np.array
        Array of size :math:`N \\times B`, where element :math:`p_{nb}`
        indicates the transmit power level of base station :math:`b` to user
        :math:`n`.

    carrier_freq : float
        Carrier frequency in Hz.

    sinr_threshold : float
        SINR threshold below which an outage occurs.

    status : list
        Binary list of length :math:`N`, where each entry indicates whether the
        corresponding user :math:`i` is active (``status[i-1] == 1``) or
        inactive (``status[i-1] == 0``).

    los : np.array, optional
        Boolean array of size :math:`N \\times B` indicates whether a
        line-of-sight (LoS) connection is established.

    bandwidth : float, optional
        Bandwidth in Hz. (Used for calculating the noise power level)

    noise_den_db : float, optional
        Noise density in dBm/Hz.


    Returns
    -------
    outage_probabilities : list
        List of length :math:`N` that contains the outage probabilities for the
        individual users.
    """
    bs_locs = np.array(bs_locs)
    user_locs = np.array(user_locs)
    carrier_freq = carrier_freq / 1e9

    num_users, num_bs = np.shape(bs_powers)
    d_3d = np.linalg.norm(
        np.expand_dims(user_locs, 1) - np.expand_dims(bs_locs, 0), axis=2
    )  # N x B

    _offset = 10 ** (-92.45 / 10)
    scale_parameters_los = _offset / (carrier_freq * d_3d / 1000) ** 2
    bs_heights = bs_locs[:, 2]
    _offset_nlos = 10 ** (-32.4 / 10)
    _scale_nlos = _offset_nlos / (
        carrier_freq**2 * d_3d ** (4.32 - 0.76 * np.log10(bs_heights))
    )
    scale_parameters_nlos = np.minimum(scale_parameters_los, _scale_nlos)

    if los is None:
        los = np.ones_like(bs_powers)
    scale_parameters = np.where(los == 1, scale_parameters_los, scale_parameters_nlos)
    scale_parameters = bs_powers * scale_parameters  # N x B

    channel_gain = np.where(los == 1, scale_parameters_los, scale_parameters_nlos)

    # TODO: WHAT IS THIS HARDCODED 8 EVERYWHERE?
    received_power = np.sum(scale_parameters * num_antennas, axis=1)  # N x B    #recieved power

    # Beamforming and Interference
    interference_wBF = np.zeros(num_users)
    interference_woBF = np.zeros(num_users)
    for i in range(num_users):
        if status[i] == 1:  # Only calculate interference for active users
            for b in range(num_bs):
                for j in range(num_users):
                    if (
                        j != i and status[j] == 1
                    ):  # Only consider active interfering users
                        # i is the main users
                        # j is the user experiencing the interference
                        theta_i = calculate_theta(user_locs[i], bs_locs[b])
                        theta_j = calculate_theta(user_locs[j], bs_locs[b])
                        phi_ij = calculate_phi(user_locs[i], bs_locs[b], user_locs[j])
                        
                        if phi_ij >= 180:
                            syn = 0
                        else:
                            syn = calculate_SyN(num_antennas, np.deg2rad(phi_ij))

                        if np.abs(np.rad2deg(theta_i)-np.rad2deg(theta_j)) >= 90:
                            szm=0
                        else:
                            szm = calculate_SzM(num_antennas, theta_j, theta_i)
                       
                        interference_wBF[j] += (
                            bs_powers[i, b]
                            * channel_gain[j, b]
                            * num_antennas  
                            * np.abs(szm)
                            * np.abs(syn)
                        )
                       
                        interference_woBF[j] += bs_powers[i, b] * channel_gain[j, b] * num_antennas

   
    noise_den = 10**(noise_den_db/10.)
    noise_power = noise_den*bandwidth

    # Outage probability
    cdf = []
    for user_params, _interference in zip(scale_parameters, interference_wBF):
        _sinr_threshold = sinr_threshold * ( noise_power +_interference)

        num_summands = len(user_params)
        user_rate_params = 1.0 / user_params
        _matrix = np.tile(user_rate_params, (num_summands, 1))
        _diff_matrix = _matrix - _matrix.T
        with np.errstate(divide="ignore"):
            _ai_matrix = np.log(_matrix) - np.log(_diff_matrix + 0j)
        np.fill_diagonal(_ai_matrix, 0)
        const_ai = np.sum(_ai_matrix, axis=1)
        log_sf = special.logsumexp(const_ai - user_rate_params * _sinr_threshold)
        log_sf = np.real(log_sf)
        # print(log_sf)
        _cdf = 1.0 - np.exp(log_sf)
        cdf.append(_cdf)
    cdf = np.maximum(cdf, np.finfo(float).eps)
    return np.ravel(cdf)


if __name__ == "__main__":
    np.random.seed(20230623)
    #uav_locations = [
    #    [10, 10, 50],
    #    [400, 400, 50],
    #    [250, 250, 30],
    #    [200, 400, 45],
    #]
    uav_locations = [
        [10, 10, 90],
        [450, 450, 90],
        [340, 40, 90],
        [250, 380, 90],
        [10, 400, 90]
    ]
    bs_locations = [[200, 200, 25], [300, 300, 25],[10, 300, 25],[450, 200, 25]]
    
    num_users = len(uav_locations)
    num_bs = len(bs_locations)
    bs_powers_db = 10*np.ones((num_users, num_bs))

    status = np.ones(num_users)
    los = np.random.randint(0, 2, size=np.shape(bs_powers_db))
    #print(los)

    bs_powers = 10 ** (np.array(bs_powers_db) / 10)
    #print("bs_power:",bs_powers)
    freq = 2.4e9
    # for sinr_threshold_db in [-110, -100, -90, -80, -70]:
    #for sinr_threshold_db in [-120]:
    for sinr_threshold_db in [-5]:
        # print(f"SNR threshold: {sinr_threshold_db:.1f}dB")
        sinr_threshold = 10 ** (sinr_threshold_db / 10)
        # path_loss = calculate_path_loss(uav_locations, bs_locations, freq)
        # print(path_loss)
        outage_prob = calculate_outage_probabilities(
            uav_locations, bs_locations, bs_powers, freq, sinr_threshold, status, los=los
        )
        print(f"Outage probability:\n{outage_prob}")
