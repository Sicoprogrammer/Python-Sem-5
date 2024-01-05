import numpy as np
import matplotlib.pyplot as plt

def simple_pendulum(theta_0, length, g, time_step, total_time):

    times = np.arange(0, total_time, time_step)
    angles = []

    theta = theta_0
    omega = 0  

    for t in times:
        
        alpha = -g / length * np.sin(theta)
        omega += alpha * time_step
        theta += omega * time_step

        angles.append(theta)

    return times, angles

# Simulation parameters
initial_angle = np.radians(30)  
pendulum_length = 1.0  
gravity = 9.8  
time_step_size = 0.01  
total_simulation_time = 5.0  

# Simulate the simple pendulum
times, angles = simple_pendulum(initial_angle, pendulum_length, gravity, time_step_size, total_simulation_time)

# Plotting the results
plt.plot(times, np.degrees(angles))  
plt.title('Simple Pendulum Motion')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.grid(True)
plt.show()
