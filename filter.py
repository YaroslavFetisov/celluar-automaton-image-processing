import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr



def add_salt_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    """Додавання шуму "сіль і перець" до зображення"""
    noisy = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))

    # Додавання "солі" (білі пікселі)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1]] = 250  # Змінимо білий колір на 250 (замість 255)

    # Додавання "перцю" (чорні пікселі)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy


def cellular_automaton_filter(image, extreme_values=(0, 250)):
    """Модифікований фільтр клітинних автоматів з виключенням екстремальних значень"""
    filtered = np.copy(image)
    padded = np.pad(image, pad_width=1, mode='edge')
    min_val, max_val = extreme_values

    for i in range(1, padded.shape[0] - 1):
        for j in range(1, padded.shape[1] - 1):
            # Отримання сусідства Мура 3x3
            neighborhood = padded[i - 1:i + 2, j - 1:j + 2].flatten()

            # Виключення пікселів з екстремальними значеннями (0 або 250)
            filtered_neighborhood = [x for x in neighborhood if x not in (min_val, max_val)]

            # Якщо всі значення були екстремальними, залишаємо оригінальне сусідство
            if len(filtered_neighborhood) == 0:
                filtered_neighborhood = neighborhood

            # Сортування значень
            sorted_neighborhood = np.sort(filtered_neighborhood)

            # Якщо після фільтрації залишилося >= 3 значення, виключаємо мінімум і максимум
            if len(sorted_neighborhood) >= 3:
                trimmed = sorted_neighborhood[1:-1]
            else:
                trimmed = sorted_neighborhood

            # Оновлення пікселя
            filtered[i - 1, j - 1] = trimmed[0]

    return filtered.astype(np.uint8)


def compare_methods_with_psnr_ssim(original, noisy):
    """Порівняння різних методів фільтрації з модифікованим алгоритмом, використовуючи PSNR та SSIM"""
    # Фільтрація модифікованим методом клітинних автоматів
    ca_modified = cellular_automaton_filter(noisy)

    # Гаусівський фільтр
    gaussian_filtered = cv2.GaussianBlur(noisy, (3, 3), 0)

    # Медіанний фільтр
    median_filtered = cv2.medianBlur(noisy, 3)

    ssim_noisy = ssim(original, noisy)
    ssim_ca_mod = ssim(original, ca_modified)
    ssim_gaussian = ssim(original, gaussian_filtered)
    ssim_median = ssim(original, median_filtered)

    # Виведення результатів
    print(f"Порівняння з еталоном:")
    print(f"- Зашумлене зображення: SSIM={ssim_noisy:.2f}")
    print(f"- Метод КА: SSIM={ssim_ca_mod:.2f}")
    print(f"- Гаусівський фільтр: SSIM={ssim_gaussian:.2f}")
    print(f"- Медіанний фільтр: SSIM={ssim_median:.2f}")

    # Візуалізація результатів
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Оригінальне зображення')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title(f'Зашумлене зображення\nSSIM: {ssim_noisy:.2f}')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(ca_modified, cmap='gray')
    plt.title(f'Метод КА\nSSIM: {ssim_ca_mod:.2f}')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(gaussian_filtered, cmap='gray')
    plt.title(f'Гаусівський фільтр\nSSIM: {ssim_gaussian:.2f}')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(median_filtered, cmap='gray')
    plt.title(f'Медіанний фільтр\nSSIM: {ssim_median:.2f}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return {
        'noisy': {'SSIM': ssim_noisy},
        'ca_modified': {'SSIM': ssim_ca_mod},
        'gaussian': {'SSIM': ssim_gaussian},
        'median': {'SSIM': ssim_median}
    }


def main():
    # Завантаження зображення
    image_path = "C:\\Users\\R3ap3r\\Downloads\\111.jpg"
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original_image is None:
        print("Не вдалося завантажити зображення. Перевірте шлях.")
        return

    # Додавання шуму "сіль (250) і перець (0)"
    noisy_image = add_salt_pepper_noise(original_image, amount=0.01)

    # Порівняння методів фільтрації
    results = compare_methods_with_psnr_ssim(original_image, noisy_image)
    print(results)

if __name__ == "__main__":
    main()