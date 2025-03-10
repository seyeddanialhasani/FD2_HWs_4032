import numpy as np
import math


def first(V, phi_deg):
    phi = math.radians(phi_deg)
    g = 9.81
    R = (V**2) / (g * math.tan(phi))
    omega_inertial = V / R
    omega_body = [0, omega_inertial * math.sin(phi), omega_inertial * math.cos(phi)]
    return {
        "Angular_velocity_in_inertial_frame": [omega_inertial],
        "Angular_velocity_in_body_frame": omega_body,
    }


result = first(250, 60)
print(result)


def second(angular_velocity, dt=0.01, total_time=10.0):
    p, q, r = angular_velocity

    phi = 0.0
    theta = 0.0
    psi = 0.0

    time_steps = int(total_time / dt)

    phi_values = np.zeros(time_steps)
    theta_values = np.zeros(time_steps)
    psi_values = np.zeros(time_steps)

    phi_dot_values = np.zeros(time_steps)
    theta_dot_values = np.zeros(time_steps)
    psi_dot_values = np.zeros(time_steps)

    for i in range(time_steps):
        phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

        phi += phi_dot * dt
        theta += theta_dot * dt
        psi += psi_dot * dt

        phi_values[i] = phi
        theta_values[i] = theta
        psi_values[i] = psi

        phi_dot_values[i] = phi_dot
        theta_dot_values[i] = theta_dot
        psi_dot_values[i] = psi_dot

    result = {
        "angles_Euler": [
            phi_values.tolist(),
            theta_values.tolist(),
            psi_values.tolist(),
        ],
        "rate_angles_Euler": [
            phi_dot_values.tolist(),
            theta_dot_values.tolist(),
            psi_dot_values.tolist(),
        ],
    }

    return result


angular_velocity = [0.33, 0.28, 0.16]
result = second(angular_velocity)

print(result)


def third(C):
    I = np.eye(3)
    is_orthogonal = np.allclose(np.dot(C, C.T), I)
    det = np.linalg.det(C)
    return is_orthogonal and np.isclose(det, 1.0)


def rotation_matrix_to_params(C):
    if not third(C):
        return "matrix does mot have rotational matrix conditions."

    theta = np.arcsin(-C[2, 0])
    psi = np.arctan2(C[1, 0] / np.cos(theta), C[0, 0] / np.cos(theta))
    phi = np.arctan2(C[2, 1] / np.cos(theta), C[2, 2] / np.cos(theta))
    euler_angles = [np.degrees(psi), np.degrees(theta), np.degrees(phi)]

    q0 = 0.5 * np.sqrt(1 + C[0, 0] + C[1, 1] + C[2, 2])
    q1 = (C[1, 2] - C[2, 1]) / (4 * q0)
    q2 = (C[2, 0] - C[0, 2]) / (4 * q0)
    q3 = (C[0, 1] - C[1, 0]) / (4 * q0)
    quaternion = [q0, q1, q2, q3]

    rotation_vector = [psi, theta, phi]

    return {
        "Euler_angles": euler_angles,
        "Quaternion_vector": quaternion,
        "Rotation_vector": rotation_vector,
    }


C = np.array(
    [[0.2802, 0.1387, 0.9499], [0.1962, 0.9603, -0.1981], [-0.9397, 0.2418, 0.2418]]
)
result = rotation_matrix_to_params(C)
print(result)


def bonus(omega, phase):
    omega_x, omega_y, omega_z = omega
    Omega = np.array(
        [[0, -omega_z, omega_y], [omega_z, 0, -omega_x], [-omega_y, omega_x, 0]]
    )

    if phase == "Cruise":
        # Cruise
        return Omega
    elif phase == "Pull-up":
        # Pull-up
        return Omega
    elif phase == "Coordinated-turn":
        # Coordinated-turn
        return Omega
    else:
        return "not a valid phase"
