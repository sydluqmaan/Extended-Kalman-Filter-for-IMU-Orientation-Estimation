import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('imu_log_simulated.csv')

def updateTime(timeStep, previous_time):

    current_time = df.at[timeStep, 'millis']   
    dt = (current_time - previous_time)/1000.0  # Convert milliseconds to seconds
    previous_time = current_time

    return dt, current_time

def updateControlInput(timeStep):
    gx = df.at[timeStep, 'gx'] 
    gy = df.at[timeStep, 'gy']
    gz = df.at[timeStep, 'gz']
    return np.array([gx, gy, gz])

def predictState(stateVector, controlInput, dt):

    def quaternionPropogationMatrix(angularVelocity):
        return np.array([[0, -angularVelocity[0], -angularVelocity[1], -angularVelocity[2]],
                        [angularVelocity[0], 0, angularVelocity[2], -angularVelocity[1]],
                        [angularVelocity[1], -angularVelocity[2], 0, angularVelocity[0]],
                        [angularVelocity[2], angularVelocity[1], -angularVelocity[0], 0]])

    attitude_quaternion = stateVector[0:4]
    bias = stateVector[4:7]

    corrected_control_input = controlInput - bias  # Remove bias from control input

    attitude_predicted = (np.identity(4) + ((dt/2) * quaternionPropogationMatrix(corrected_control_input))) @ attitude_quaternion
    attitude_predicted = attitude_predicted / np.linalg.norm(attitude_predicted)  # Normalize quaternion

    return np.concatenate((attitude_predicted, bias))

def predictCovariance(stateVector, covariance, controlInput, dt, Q):

    def numerical_jacobian_F(state, control_input, dt, epsilon=1e-6):
        state = np.array(state, dtype=float)
        f0 = predictState(state, control_input, dt)
        n = state.size
        m = f0.size
        J = np.zeros((m, n))

        for i in range(n):
            dx = np.zeros(n)
            dx[i] = epsilon
            f1 = predictState(state + dx, control_input, dt)
            J[:, i] = (f1 - f0) / epsilon

        return J

    F = numerical_jacobian_F(stateVector, controlInput, dt)
    return F @ covariance @ F.T + Q

def readAccelerometerMeasurement(timeStep):
    ax = df.at[timeStep, 'ax'] 
    ay = df.at[timeStep, 'ay']
    az = df.at[timeStep, 'az']
    accel_reading = np.array([ax, ay, az])
    norm = np.linalg.norm(accel_reading)
    if norm == 0:
        return np.array([0, 0, 0])
        
    normalized_accel = accel_reading / norm
    return normalized_accel

def updateStateAndCovariance(timeStep, stateVector, P, R):
    
    def quat_to_rotmat(q):
        x, y, z, w = q
        return np.array([
            [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ])

    def predictAccelerometerMeasurement(state):
        q = state[0:4]
        R = quat_to_rotmat(q)
        g_ref = np.array([0, 0, 1])
        return R.T @ g_ref
    
    def numerical_jacobian_H(x, epsilon=1e-6):
        h0 = predictAccelerometerMeasurement(x)
        n = len(x)
        m = len(h0)
        J = np.zeros((m, n))
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = epsilon
            J[:, i] = (predictAccelerometerMeasurement(x + dx) - h0) / epsilon
        return J

    z = readAccelerometerMeasurement(timeStep)
    z_predicted = predictAccelerometerMeasurement(stateVector)
    residual = z - z_predicted

    H = numerical_jacobian_H(stateVector)
    residual_covariance = H @ P @ H.T + R         # R = Sensor noise covariance matrix

    kalman_gain = P @ H.T @ np.linalg.inv(residual_covariance)

    updated_state_Vector = stateVector + (kalman_gain @ residual)           # Update state vector with corrected values
    updated_covariance = (np.eye(len(stateVector)) - kalman_gain @ H) @ P   # Update covariance

    return updated_state_Vector, updated_covariance

def stepEKF(stateVector, control_input, covariance, dt, Q, currentTimeStep, accelerometer_noise):
    state_predicted = predictState(stateVector, control_input, dt)
    covariance_predicted = predictCovariance(stateVector, covariance, control_input, dt, Q)

    state_vector_updated, covariance_updated = updateStateAndCovariance(currentTimeStep, state_predicted, covariance_predicted, accelerometer_noise)

    return state_vector_updated, covariance_updated

def quaternion_to_euler(q):
    x, y, z, w = q

    # Roll (x)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)  # 90 deg if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(np.array([roll, pitch, yaw]))

def plot_angle_history(angle_log):
    data = np.array(angle_log)
    timestamps = data[:, 0] - data[0, 0]
    roll = data[:, 1]
    pitch = data[:, 2]
    yaw = data[:, 3]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, roll, label='Roll', color='r')
    plt.plot(timestamps, pitch, label='Pitch', color='g')
    plt.plot(timestamps, yaw, label='Yaw', color='b')

    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.title('EKF Output Angles')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calibrateGyroBias(df):
    x_average = (df['gx'][:500].sum())/500
    y_average = (df['gy'][:500].sum())/500
    z_average = (df['gz'][:500].sum())/500

    return np.array([x_average, y_average, z_average])

def main():  
    stateVector = np.array([0, 0, 0, 1, 0, 0, 0])                                   # Initialize state vector: [q0, q1, q2, q3, bx, by, bz]

    q_noise = 1e-6                                                                  # Gyro noise
    b_noise = 1e-7                                                                  # Gyro bias
    Q = np.diag([q_noise, q_noise, q_noise, q_noise, b_noise, b_noise, b_noise])    # Process noise covariance matrix

    P = np.zeros((7,7))                                                             # Initialize state covariance matrix
    P[0:4, 0:4] = np.eye(4) * 1e-4                                                  # Initial covariance for attitude
    P[4:7, 4:7] = np.eye(3) * 1e-3                                                  # Initial covariance for bias
    accelerometer_noise = np.diag([1e-3, 1e-3, 1e-3])

    previous_time = df.at[0, 'millis']                                              # Initialize previous time
    currentTimeStep = 1                                                             # Start from the second time step
    dt = 0.0                                                                        # Initialize time delta

    gyroscope_bias = calibrateGyroBias(df)                                          # Calculate gyro bias
    stateVector[4:7] = gyroscope_bias

    angle_history = []                                                              # Initialize angle history for plotting

    while currentTimeStep < len(df):
        dt, previous_time = updateTime(currentTimeStep, previous_time)
        control_input = updateControlInput(currentTimeStep)
        stateVector, P = stepEKF(stateVector, control_input, P, dt, Q, currentTimeStep, accelerometer_noise)

        angles = quaternion_to_euler(stateVector[0:4])
        timestamp = df.at[currentTimeStep, 'millis'] / 1000.0
        angle_history.append([timestamp]+angles.tolist())

        currentTimeStep += 1

    plot_angle_history(angle_history)

main()