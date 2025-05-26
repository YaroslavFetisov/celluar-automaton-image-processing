import numpy as np
import cupy as cp
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import time
from config import IMAGE_PATH_FILTER, NOISE_AMOUNT

# CUDA ядро для фільтрації клітинного автомата
apply_ca_filter_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void applyCAFilter(const unsigned char* image, unsigned char* output, 
                       int width, int height, unsigned char min_val, unsigned char max_val) {
        int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

        if (x >= width + 1 || y >= height + 1) return;

        int idx = (y - 1) * width + (x - 1);
        unsigned char neighborhood[9];
        int count = 0;

        // Зчитування околиці 3x3
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int n_idx = (y + dy) * (width + 2) + (x + dx);
                if (image[n_idx] != min_val && image[n_idx] != max_val) {
                    neighborhood[count++] = image[n_idx];
                }
            }
        }

        // Сортування сусідів
        for (int i = 0; i < count - 1; i++) {
            for (int j = i + 1; j < count; j++) {
                if (neighborhood[i] > neighborhood[j]) {
                    unsigned char temp = neighborhood[i];
                    neighborhood[i] = neighborhood[j];
                    neighborhood[j] = temp;
                }
            }
        }

        // Вибір значення
        output[idx] = (count >= 3) ? neighborhood[1] : (count > 0 ? neighborhood[0] : image[y * (width + 2) + x]);
    }
    ''', 'applyCAFilter')

def add_salt_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    noisy = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1]] = 250
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def cellular_automaton_filter(image, extreme_values=(0, 250)):
    start_time = time.time()
    image_gpu = cp.array(image, dtype=cp.uint8)
    height, width = image.shape
    padded = cp.pad(image_gpu, pad_width=1, mode='edge')
    output_gpu = cp.zeros((height, width), dtype=cp.uint8)

    block_size = (32, 8)
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])

    apply_ca_filter_kernel(grid_size, block_size,
                          (padded, output_gpu, width, height, extreme_values[0], extreme_values[1]))

    result = cp.asnumpy(output_gpu).astype(np.uint8)
    end_time = time.time()
    print(f"Час виконання алгоритму КА: {end_time - start_time:.4f} секунд")
    return result

def compare_methods_with_psnr_ssim(original, noisy):
    ca_modified = cellular_automaton_filter(noisy)
    gaussian_filtered = cv2.GaussianBlur(noisy, (3, 3), 0)
    median_filtered = cv2.medianBlur(noisy, 3)

    ssim_noisy = ssim(original, noisy)
    ssim_ca_mod = ssim(original, ca_modified)
    ssim_gaussian = ssim(original, gaussian_filtered)
    ssim_median = ssim(original, median_filtered)

    print(f"Порівняння з еталоном:")
    print(f"- Зашумлене зображення: SSIM={ssim_noisy:.2f}")
    print(f"- Метод КА: SSIM={ssim_ca_mod:.2f}")
    print(f"- Гаусівський фільтр: SSIM={ssim_gaussian:.2f}")
    print(f"- Медіанний фільтр: SSIM={ssim_median:.2f}")

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
    image_path = IMAGE_PATH_FILTER
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print("Не вдалося завантажити зображення. Перевірте шлях.")
        return
    noisy_image = add_salt_pepper_noise(original_image, amount=NOISE_AMOUNT)
    results = compare_methods_with_psnr_ssim(original_image, noisy_image)
    print(results)

if __name__ == "__main__":
    main()