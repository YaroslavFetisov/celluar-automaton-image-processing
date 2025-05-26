import matplotlib.pyplot as plt

# Data for the graph
image_sizes = ['128x128', '256x256', '512x512', '1024x1024', '2048x2048']
cpu_times = [0.0448, 0.1775, 0.7115, 2.8830, 11.4346]
gpu_times = [0.1468, 0.1563, 0.1472, 0.1783, 0.1677]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(image_sizes, cpu_times, label='CPU', marker='o', linestyle='-', color='blue')
plt.plot(image_sizes, gpu_times, label='GPU', marker='o', linestyle='-', color='red')

# Add labels and title
plt.xlabel('Розмір зображення')
plt.ylabel('Час виконання (секунди)')
plt.title('Час виконання на CPU та GPU для різних розмірів зображень')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
