# EKF for IMU Orientation Estimation
This project implements an Extended Kalman Filter (EKF) in Python to estimate orientation using accelerometer and gyroscope measurements. Orientation is represented with quaternions.

## Features
- Implements initial gyro bias calibration.
- Uses a 7 element state vector. 4 element quaternion for orientation and 3 element gyro bias.
- Estimates roll and pitch angles.
- Uses accelerometer to correct gryo drift with sensor fusion.
  
## Note
The current implementation does not include magnetometer integration for yaw drift correction. As a result, it accurately estimates roll and pitch, but yaw may drift over time.

## Example
Demonstration using MPU6500 data.

### Accelerometer Data
<img width="1920" height="967" alt="Accelerometer Data" src="https://github.com/user-attachments/assets/620dc71b-7132-4cb2-863e-c87832ba71d9" />

### Gyro Data
<img width="1920" height="967" alt="Gyro Data" src="https://github.com/user-attachments/assets/867eaab9-7dc4-4e78-983a-7c4774883a4f" />

### EKF Output
<img width="1920" height="967" alt="EKF Output" src="https://github.com/user-attachments/assets/508f6648-da3e-4d5e-8013-52943773fdc0" />
