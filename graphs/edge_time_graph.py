from matplotlib import pyplot as plt

image_sizes = ['128x128', '256x256', '512x512', '1024x1024', '2048x2048']
cpu_times_edge_corrected = [2.2944, 3.5475, 11.5804, 54.5078, 199.7777]
gpu_times_edge_corrected = [3.0435, 3.0872, 3.9506, 4.2090, 5.1271]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(image_sizes, cpu_times_edge_corrected, label='CPU', marker='o', linestyle='-', color='blue')
plt.plot(image_sizes, gpu_times_edge_corrected, label='GPU', marker='o', linestyle='-', color='red')

# Add labels and title
plt.xlabel('Розмір зображення')
plt.ylabel('Час виконання (секунди)')
plt.title('Час виконання на CPU та GPU для знаходження країв')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
