import numpy as np


def calculate_theta(user_locs, bs_locations):
    a = np.sqrt(
        (user_locs[0] - bs_locations[0]) ** 2 + (user_locs[1] - bs_locations[1]) ** 2
    )
    b = user_locs[2] - bs_locations[2]
    theta = np.arctan(a / b)
    return theta


def calculate_phi(user_1, bs_loc, user_2):
    vector_1b = user_1 - bs_loc
    vector_2b = user_2 - bs_loc

    dot_product = np.dot(vector_1b, vector_2b)
    magnitude_1b = np.linalg.norm(vector_1b)
    magnitude_b2 = np.linalg.norm(vector_2b)

    angle_radians = np.arccos(dot_product / (magnitude_1b * magnitude_b2))
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def calculate_SzM(M, theta, theta_target):
    SzM = 0
    for m in range(1, M + 1):
        term = np.exp(
            1j * (m - 1) * (np.pi * np.sin(theta - theta_target))
        )
        SzM += term
    return SzM / M


def calculate_SyN(N, phi_ij):
    SyN = 0
    for n in range(1, N + 1):
        term = np.exp(1j * (n - 1) * (np.pi * (np.sin(phi_ij))))
        SyN += term
    return SyN/N
