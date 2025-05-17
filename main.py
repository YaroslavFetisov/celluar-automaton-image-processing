import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


# Функція для ініціалізації клітинного автомата
def initialize_automaton(image, rule):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_image = gray_image / 255.0
    automaton = np.zeros_like(normalized_image)
    return normalized_image, automaton


# Функція для застосування правил клітинного автомата з використанням градієнта
def apply_automaton_rule(automaton, normalized_image, rule):
    threshold = rule[0]  # Поріг для активації
    size = rule[1]  # Розмір області перевірки

    # Обчислюємо градієнт інтенсивності
    grad_x = cv2.Sobel(normalized_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)

    for i in range(1, automaton.shape[0] - 1):
        for j in range(1, automaton.shape[1] - 1):
            # Якщо градієнт вищий за поріг, активуємо клітинку
            if gradient[i, j] > threshold:
                automaton[i, j] = 1
            else:
                automaton[i, j] = 0
    return automaton


# Обчислення контурів за допомогою Canny
def detect_edges_with_canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return edges


# Функція для обчислення метрики схожості (середньоквадратична похибка)
def compute_similarity(automaton, edges):
    similarity = np.sqrt(np.sum((automaton - edges) ** 2) / automaton.size)
    return similarity


# Генерація початкової популяції правил
def generate_population(pop_size, rule_range):
    population = []
    for _ in range(pop_size):
        threshold = random.uniform(rule_range[0], rule_range[1])
        size = random.randint(1, 3)  # Розмір області перевірки
        population.append([threshold, size])
    return population


# Вибірка найкращих хромосом (найменша похибка)
def select_best(population, image, edges):
    fitness = []
    for rule in population:
        normalized_image, automaton = initialize_automaton(image, rule)
        automaton = apply_automaton_rule(automaton, normalized_image, rule)
        similarity = compute_similarity(automaton, edges)
        fitness.append((rule, similarity))

    # Сортуємо за похибкою (найменша похибка - найкращий)
    fitness.sort(key=lambda x: x[1])

    # Перевіряємо, чи є достатньо хромосом, і повертаємо найкращі
    best_individuals = fitness[:len(fitness) // 2]
    if len(best_individuals) == 0:
        # Якщо немає достатньо хромосом, повертаємо хоча б одну випадкову
        best_individuals = [fitness[0]]

    return best_individuals


# Кросовер (змішування двох хромосом)
def crossover(parent1, parent2):
    threshold = (parent1[0] + parent2[0]) / 2
    size = random.choice([parent1[1], parent2[1]])
    return [threshold, size]


# Мутація (зміна одного параметра)
def mutate(rule, mutation_rate, rule_range):
    if random.random() < mutation_rate:
        rule[0] = random.uniform(rule_range[0], rule_range[1])
    if random.random() < mutation_rate:
        rule[1] = random.randint(1, 3)
    return rule


# Головна функція для еволюції популяції з урахуванням непарної кількості
def genetic_algorithm(image, generations=1000, pop_size=50, mutation_rate=0.01, rule_range=(0.3, 0.7)):
    # Генерація початкової популяції
    edges = detect_edges_with_canny(image)
    population = generate_population(pop_size, rule_range)

    # Запуск генетичного алгоритму
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")

        # Оцінка популяції
        best_individuals = select_best(population, image, edges)

        # Створення нової популяції
        new_population = []

        # Якщо непарна кількість найкращих, одну залишаємо без пари
        if len(best_individuals) % 2 == 1:
            new_population.append(best_individuals[-1][0])  # Додаємо останню хромосому без пари

        for i in range(0, len(best_individuals) - 1, 2):
            parent1, parent2 = best_individuals[i][0], best_individuals[i + 1][0]
            child = crossover(parent1, parent2)
            new_population.append(child)

        # Мутація нових хромосом
        new_population = [mutate(rule, mutation_rate, rule_range) for rule in new_population]

        # Оновлення популяції
        population = new_population

    # Повертаємо найкраще правило після всіх поколінь
    best_rule = select_best(population, image, edges)[0][0]
    return best_rule


# Головна функція для запуску
def main(image_path):
    image = cv2.imread(image_path)

    # Запуск генетичного алгоритму для еволюції правил клітинного автомата
    best_rule = genetic_algorithm(image)
    print(f"Best rule found: Threshold = {best_rule[0]}, Size = {best_rule[1]}")

    # Отримуємо автомат з найкращим правилом
    normalized_image, automaton = initialize_automaton(image, best_rule)
    automaton = apply_automaton_rule(automaton, normalized_image, best_rule)

    # Обчислюємо контури Canny
    edges = detect_edges_with_canny(image)

    # Обчислюємо точність
    accuracy = compute_similarity(automaton, edges)
    print(f"Accuracy (Mean Squared Error): {accuracy}")

    # Виведення результатів
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Automaton Contours")
    plt.imshow(automaton, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Canny Edges")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.show()


# Запуск програми
if __name__ == "__main__":
    image_path = "C:\\Users\\R3ap3r\\Downloads\\2.png"  # Заміни на шлях до твого зображення
    main(image_path)
