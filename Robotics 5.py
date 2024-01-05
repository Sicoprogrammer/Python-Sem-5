import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import trim_mean

def triangular_mf(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

def fuzzy_inference(temperature, humidity):
    # Fuzzification
    temp_cold = triangular_mf(temperature, 10, 25, 40)
    temp_normal = triangular_mf(temperature, 20, 35, 50)
    temp_hot = triangular_mf(temperature, 40, 60, 75)

    hum_dry = triangular_mf(humidity, 10, 25, 40)
    hum_comfortable = triangular_mf(humidity, 30, 50, 70)
    hum_humid = triangular_mf(humidity, 60, 75, 90)

    # Rule Evaluation
    rule1 = min(temp_cold, hum_dry)
    rule2 = min(temp_normal, hum_comfortable)
    rule3 = min(temp_hot, hum_humid)
    rule4 = min(temp_hot, hum_humid)

    # Aggregation
    aggregated = np.max([rule1, rule2, rule3, rule4])

    # Defuzzification (Centroid Method)
    fan_speed_values = np.array([20, 40, 60, 80, 100, 120])
    centroid = np.sum(fan_speed_values * aggregated) / np.sum(aggregated)

    return centroid

# Example inputs
temperature_input = 43  # Celsius
humidity_input = 50  # Percentage

# Fuzzy inference
fan_speed_output = fuzzy_inference(temperature_input, humidity_input)

# Display results
print(f"Temperature: {temperature_input}°C, Humidity: {humidity_input}%")
print(f"Recommended Fan Speed: {fan_speed_output:.2f}")

# Plot membership functions for visualization
temperature_range = np.arange(0, 80, 0.1)
humidity_range = np.arange(0, 100, 0.1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(temperature_range, triangular_mf(temperature_range, 10, 25, 40), label='Cold')
plt.plot(temperature_range, triangular_mf(temperature_range, 20, 35, 50), label='Normal')
plt.plot(temperature_range, triangular_mf(temperature_range, 40, 60, 75), label='Hot')
plt.title('Temperature Membership Functions')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(humidity_range, triangular_mf(humidity_range, 10, 25, 40), label='Dry')
plt.plot(humidity_range, triangular_mf(humidity_range, 30, 50, 70), label='Comfortable')
plt.plot(humidity_range, triangular_mf(humidity_range, 60, 75, 90), label='Humid')
plt.title('Humidity Membership Functions')
plt.legend()

plt.show()
print("Done")