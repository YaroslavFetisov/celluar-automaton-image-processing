import matplotlib.pyplot as plt

# Дані
noise_levels = [10, 25, 50, 75, 90, 99]
ca_ssim = [0.87, 0.87, 0.87, 0.81, 0.72, 0.66]
gaussian_ssim = [0.49, 0.32, 0.21, 0.15, 0.13, 0.12]
median_ssim = [0.93, 0.86, 0.50, 0.19, 0.13, 0.11]

# Створення графіка
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, ca_ssim, 'b-o', label='Метод КА', linewidth=2, markersize=8)
plt.plot(noise_levels, gaussian_ssim, 'r--s', label='Гаусівський фільтр', linewidth=2, markersize=8)
plt.plot(noise_levels, median_ssim, 'g-.D', label='Медіанний фільтр', linewidth=2, markersize=8)

# Налаштування графіка
plt.title('Порівняння ефективності методів фільтрації', fontsize=14, fontweight='bold')
plt.xlabel('Рівень шуму (%)', fontsize=12)
plt.ylabel('Значення SSIM', fontsize=12)
plt.xticks(noise_levels)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.ylim(0, 1)  # Обмеження осі Y від 0 до 1

# Додаткові підписи
for i, (ca, gauss, med) in enumerate(zip(ca_ssim, gaussian_ssim, median_ssim)):
    plt.text(noise_levels[i], ca+0.02, f'{ca:.2f}', ha='center', color='blue')
    plt.text(noise_levels[i], gauss-0.05, f'{gauss:.2f}', ha='center', color='red')
    plt.text(noise_levels[i], med-0.05, f'{med:.2f}', ha='center', color='green')

plt.tight_layout()
plt.show()