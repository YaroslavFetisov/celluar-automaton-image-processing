from cellular_automaton_edge_detector import CellularAutomatonEdgeDetector
from range_optimizer import RangeOptimizer
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Крок 1: Знаходимо оптимальне правило 3x3 (як раніше)
    detector = CellularAutomatonEdgeDetector(
        "C:\\Users\\R3ap3r\\Downloads\\2.png",
        population_size=1000,
        max_generations=50,
        early_stopping=10,
        n_jobs=-1
    )
    best_3x3_rule, history = detector.evolve()

    # Крок 2: Оптимізуємо діапазон для 7x7
    optimizer = RangeOptimizer(detector.image_processor, best_3x3_rule)
    lower, upper, f1 = optimizer.optimize_range()

    # Фінальний результат
    edges_combined = optimizer.apply_range_rule(detector.image_processor.binary_image, lower, upper)

    # Візуалізація
    plt.figure(figsize=(15, 5))
    plt.subplot(141), plt.imshow(detector.image_processor.original_image, cmap='gray'), plt.title("Оригінал")
    plt.subplot(142), plt.imshow(detector.image_processor.canny_edges, cmap='gray'), plt.title("Кенні")
    plt.subplot(143), plt.imshow(detector.detect_edges(best_3x3_rule), cmap='gray'), plt.title("CA-ED (3x3)")
    plt.subplot(144), plt.imshow(edges_combined, cmap='gray'), plt.title(f"CA-ED (3x3 + 7x7)\nДіапазон: [{lower}, {upper}]")
    plt.show()

    print(f"Найкраще правило 3x3: {best_3x3_rule}")
    print(f"Найкращий діапазон для 7x7: [{lower}, {upper}], F1: {f1:.4f}")