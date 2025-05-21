import os
import cv2
from cellular_automaton_edge_detector import CellularAutomatonEdgeDetector
from filter import cellular_automaton_filter
from range_optimizer import RangeOptimizer
import matplotlib.pyplot as plt
import config

if __name__ == "__main__":
    image_path = config.IMAGE_PATH_EDGE

    # Крок 1: Фільтрація від шуму
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    filtered_image = cellular_automaton_filter(original_image)

    # Крок 2: Зберігаємо відфільтроване зображення в папку filtered
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # Отримуємо ім'я файлу без розширення
    output_path = os.path.join('filtered', f"{base_name}_filtered.png")  # Шлях до папки filtered

    cv2.imwrite(output_path, filtered_image)

    # Крок 2: Знаходимо оптимальне правило 3x3
    detector = CellularAutomatonEdgeDetector(
        image_path,
        population_size=config.POPULATION_SIZE,
        max_generations=config.MAX_GENERATIONS,
        early_stopping=config.EARLY_STOPPING,
        mutation_rate=config.MUTATION_RATE,
        n_jobs=config.N_JOBS
    )
    best_3x3_rule, history = detector.evolve()

    # Крок 3: Оптимізуємо діапазон для 7x7
    optimizer = RangeOptimizer(detector.image_processor, best_3x3_rule)
    lower, upper, f1 = optimizer.optimize_range()

    # Фінальний результат
    edges_combined = optimizer.apply_range_rule(detector.image_processor.binary_image, lower, upper)

    # Візуалізація
    plt.figure(figsize=(15, 5))
    plt.subplot(141), plt.imshow(original_image, cmap='gray'), plt.title("Оригінал")
    plt.subplot(142), plt.imshow(filtered_image, cmap='gray'), plt.title("Відфільтроване")
    plt.subplot(143), plt.imshow(detector.image_processor.canny_edges, cmap='gray'), plt.title("Кенні")
    plt.subplot(144), plt.imshow(edges_combined, cmap='gray'), plt.title(f"CA-ED (3x3 + 7x7)\nДіапазон: [{lower}, {upper}]")
    plt.show()

    print(f"Найкраще правило 3x3: {best_3x3_rule}")
    print(f"Найкращий діапазон для 7x7: [{lower}, {upper}], F1: {f1:.4f}")