import numpy as np
import cv2
import matplotlib.pyplot as plt

# Функція для перетворення зображення у відтінки сірого
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Функція для бінаризації зображення за допомогою методу Оцу
def binarize_image(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

# Функція для визначення країв за допомогою стандартного детектора Кенні
def canny_edge_detection(image):
    return cv2.Canny(image, 100, 200)

# Оцінка результату: порівнюємо з еталонними краями
def evaluate_result(predicted_edges, ground_truth_edges):
    # Оцінка за допомогою середньоквадратичної похибки (MSE)
    return np.sum((predicted_edges - ground_truth_edges) ** 2) / predicted_edges.size

# Функція для застосування одного правила до одного пікселя
def apply_rule_to_pixel(i, j, image, rule_set):
    neighbors = image[i-1:i+2, j-1:j+2]
    for idx, rule in enumerate(rule_set):
        if np.array_equal(neighbors, rule):
            return (i, j, idx + 1)  # Повертаємо піксель та правило
    return None

# Функція для застосування клітинного автомата
def cellular_automaton_edge_detection(image, rule_set):
    m, n = image.shape
    ca_image = np.copy(image)

    print("Застосування клітинного автомата для виявлення країв...")
    total_rules = len(rule_set)  # Кількість правил

    # Відслідковуємо прогрес по рядках
    for i in range(1, m-1):
        for j in range(1, n-1):
            # Визначити сусідів для поточного пікселя
            neighbors = image[i-1:i+2, j-1:j+2]

            # Застосування кожного правила для пошуку краю
            for idx, rule in enumerate(rule_set):
                if np.array_equal(neighbors, rule):
                    ca_image[i, j] = 255  # Змінюємо піксель на білий, якщо правило співпало

    return ca_image

# Генерація всіх можливих правил для виявлення країв (около Мура)
def generate_rules():
    rule_set = [
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),  # Правило 1
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),  # Правило 2
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),  # Правило 3
        np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0]]),  # Правило 4
        np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]]),  # Правило 5
        np.array([[0, 0, 1], [1, 1, 0], [0, 0, 0]]),  # Правило 6
        np.array([[0, 1, 1], [1, 1, 1], [0, 0, 0]]),  # Правило 7
        np.array([[1, 1, 0], [1, 1, 1], [0, 0, 0]]),  # Правило 8
        np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]]),  # Правило 9
        np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1]]),  # Правило 10
        np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),  # Правило 11
        np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]]),  # Правило 12
        np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0]]),  # Правило 13
        np.array([[1, 1, 1], [0, 1, 1], [1, 0, 0]]),  # Правило 14
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]]),  # Правило 15
        np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0]]),  # Правило 16
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]]),  # Правило 17
        np.array([[1, 1, 1], [0, 0, 1], [1, 0, 0]]),  # Правило 18
        np.array([[0, 0, 1], [1, 0, 1], [0, 0, 0]]),  # Правило 19
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]]),  # Правило 20
        np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]),  # Правило 21
        np.array([[1, 1, 1], [0, 1, 0], [1, 0, 0]]),  # Правило 22
        np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]]),  # Правило 23
        np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]]),  # Правило 24
        np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]]),  # Правило 25
        np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]]),  # Правило 26
        np.array([[0, 0, 1], [1, 1, 1], [1, 1, 0]]),  # Правило 27
        np.array([[0, 1, 0], [0, 1, 1], [1, 1, 0]]),  # Правило 28
        np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]]),  # Правило 29
        np.array([[1, 1, 0], [0, 1, 0], [1, 1, 0]]),  # Правило 30
        np.array([[0, 1, 1], [0, 1, 1], [1, 1, 1]]),  # Правило 31
        np.array([[0, 1, 0], [1, 0, 0], [1, 1, 1]]),  # Правило 32
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]),  # Правило 33
        np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0]]),  # Правило 34
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]]),  # Правило 35
        np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1]]),  # Правило 36
        np.array([[1, 0, 1], [1, 0, 0], [1, 1, 1]]),  # Правило 37
        np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1]]),  # Правило 38
        np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]]),  # Правило 39
        np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]]),  # Правило 40
        np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1]]),  # Правило 41
        np.array([[1, 1, 1], [0, 0, 0], [1, 1, 0]]),  # Правило 42
        np.array([[0, 1, 1], [1, 0, 0], [1, 1, 1]]),  # Правило 43
        np.array([[0, 0, 0], [1, 1, 1], [0, 1, 1]]),  # Правило 44
        np.array([[1, 1, 1], [0, 0, 1], [0, 1, 1]]),  # Правило 45
        np.array([[0, 0, 0], [1, 1, 1], [1, 0, 1]]),  # Правило 46
        np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]]),  # Правило 47
        np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1]]),  # Правило 48
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]]),  # Правило 49
        np.array([[1, 0, 0], [1, 1, 1], [1, 1, 0]]),  # Правило 50
        np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]]),  # Правило 51
    ]
    return rule_set

# Алгоритм Forward Feature Selection (FFS)
def forward_feature_selection(image, rule_set, ground_truth_edges):
    selected_rules = []  # Результуючий список для вибраних правил
    best_score = float('inf')  # Ініціалізація найкращого результату

    print("Початок алгоритму Forward Feature Selection (FFS)...")
    while len(rule_set) > 0:
        # Для кожного правила пробуємо застосувати і оцінити результат
        best_rule = None
        best_rule_score = float('inf')

        for rule in rule_set:
            # Створюємо копію зображення для кожного правила
            temp_image = np.copy(image)
            temp_image = cellular_automaton_edge_detection(temp_image, [rule])

            # Оцінка результату
            score = evaluate_result(temp_image, ground_truth_edges)
            print(f"Оцінка для правила {rule}: {score}")

            # Якщо це найкращий результат, зберігаємо правило
            if score < best_rule_score:
                best_rule_score = score
                best_rule = rule

        # Якщо результат покращився, додаємо правило до вибраного списку
        if best_rule_score < best_score:
            best_score = best_rule_score
            selected_rules.append(best_rule)
            rule_set.remove(best_rule)  # Видаляємо правило з набору

            print(f"Додано правило з найкращим результатом. Поточний набір правил: {len(selected_rules)}")
        else:
            print("Не знайдено поліпшення, припиняємо.")
            break

    return selected_rules, best_score

# Основна функція
def main(image_path):
    print("Завантаження зображення...")
    # Завантажуємо зображення
    image = cv2.imread(image_path)

    # Перетворюємо зображення у відтінки сірого
    grayscale_image = convert_to_grayscale(image)

    # Бінаризуємо зображення для порівняння
    binary_image = binarize_image(grayscale_image)

    # Отримуємо еталонне зображення країв за допомогою Кенні
    ground_truth_edges = canny_edge_detection(grayscale_image)

    # Генеруємо набір правил для КА
    rule_set = generate_rules()

    # Використовуємо Forward Feature Selection для вибору найкращих правил
    selected_rules, best_score = forward_feature_selection(binary_image, rule_set, ground_truth_edges)

    print(f"Найкращі правила: {selected_rules}")
    print(f"Найкраща оцінка результату: {best_score}")

    # Візуалізація результатів
    final_edges = cellular_automaton_edge_detection(binary_image, selected_rules)
    plt.subplot(1, 2, 1), plt.imshow(ground_truth_edges, cmap='gray'), plt.title('Еталонні краю (Кенні)')
    plt.subplot(1, 2, 2), plt.imshow(final_edges, cmap='gray'), plt.title('Краї за КА')
    plt.show()

if __name__ == "__main__":
    image_path = 'C:\\Users\\R3ap3r\\Downloads\\1.jpeg'  # Задайте шлях до вашого зображення
    main(image_path)
