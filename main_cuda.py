import cv2
from CUDA.cellular_automaton_edge_detector_cuda import CellularAutomatonEdgeDetector
from CPU.range_optimizer import RangeOptimizer
import matplotlib.pyplot as plt
import config
import numpy as np
import cupy as cp
import time

if __name__ == "__main__":
    image_path = config.IMAGE_PATH_EDGE

    # Завантажуємо оригінальне зображення
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    start_time = time.time()
    # Знаходимо оптимальне правило 3x3
    detector = CellularAutomatonEdgeDetector(
        image_path,
        population_size=config.POPULATION_SIZE,
        max_generations=config.MAX_GENERATIONS,
        early_stopping=config.EARLY_STOPPING,
        mutation_rate=config.MUTATION_RATE,
        # n_jobs=config.N_JOBS
    )
    best_3x3_rule, history = detector.evolve()
    end_time = time.time()
    print(f"Час виконання алгоритму КА: {end_time - start_time:.4f} секунд")

    # Оптимізуємо діапазон для 3x3
    optimizer = RangeOptimizer(detector.image_processor, best_3x3_rule)
    lower, upper, f1 = optimizer.optimize_range()

    # Фінальний результат
    edges_combined = optimizer.apply_range_rule(detector.image_processor.binary_image, lower, upper)

    # Переконаємося, що edges_combined є NumPy масивом
    if isinstance(edges_combined, cp.ndarray):
        edges_combined = edges_combined.get()

    # Лапласіан
    laplacian = cv2.Laplacian(original_image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    _, laplacian_binary = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)

    # Візуалізація без координат
    plt.figure(figsize=(20, 5))

    plt.subplot(141)
    plt.imshow(original_image, cmap='gray')
    plt.title("Оригінал")
    plt.axis('off')

    # Переконаємося, що canny_edges є NumPy масивом
    canny_edges = detector.image_processor.canny_edges
    if isinstance(canny_edges, cp.ndarray):
        canny_edges = canny_edges.get()

    plt.subplot(142)
    plt.imshow(canny_edges, cmap='gray')
    plt.title("Кенні")
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(laplacian_binary, cmap='gray')
    plt.title("Лапласіан")
    plt.axis('off')

    plt.subplot(144)
    plt.imshow(edges_combined, cmap='gray')
    plt.title(f"Метод КА")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Найкраще правило 3x3: {best_3x3_rule}")
    print(f"Діапазон для 3x3: [{lower}, {upper}], F1: {f1:.4f}")
