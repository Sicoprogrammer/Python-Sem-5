import numpy as np
import matplotlib.pyplot as plt

def plot_sine_wave(amplitude, frequency, label):
    time = np.linspace(0, 2 * np.pi, 1000)
    wave = amplitude * np.sin(frequency * time)
    plt.plot(time, wave, label=f'Amp={amplitude}, Freq={frequency}')

# Amplitude and frequency combinations
combinations = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]

# Plot sine waves
plt.figure(figsize=(10, 6))
for amp, freq in combinations:
    plot_sine_wave(amp, freq, f'Amp={amp}, Freq={freq}')

# Customize the plot
plt.title('Sine Waves with Different Amplitudes and Frequencies')
plt.xlabel('Time')
plt.ylabel('Amplitude * sin(Frequency * Time)')
plt.legend()
plt.grid(True)
plt.show()
