from cellular_automaton_edge_detector import CellularAutomatonEdgeDetector
import matplotlib.pyplot as plt

if __name__ == "__main__":
    detector = CellularAutomatonEdgeDetector(
        "C:\\Users\\R3ap3r\\Downloads\\2.png",
        population_size=1000,
        max_generations=50,
        early_stopping=5,
        n_jobs=-1
    )

    best_rule, history = detector.evolve()
    edges = detector.detect_edges(best_rule)

    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(detector.image_processor.original_image, cmap='gray'), plt.title("Оригінал")
    plt.subplot(132), plt.imshow(detector.image_processor.canny_edges, cmap='gray'), plt.title("Кенні")
    plt.subplot(133), plt.imshow(edges, cmap='gray'), plt.title("CA-ED")
    plt.show()

    plt.plot(history)
    plt.title("Прогрес F1-міри"), plt.xlabel("Покоління"), plt.ylabel("Найкраща F1-міра")
    plt.show()

    print(f"Найкраще правило: {best_rule}")
