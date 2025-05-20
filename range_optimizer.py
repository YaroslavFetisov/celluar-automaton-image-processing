import cv2
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

class RangeOptimizer:
    def __init__(self, image_processor, best_3x3_rule):
        self.image_processor = image_processor
        self.best_3x3_rule = best_3x3_rule
        self.original_image = image_processor.original_image
        self.binary_image = image_processor.binary_image
        self.canny_flat = image_processor.canny_flat

    def apply_range_rule(self, image, lower, upper):
        pad = 3  # Радіус 7x7 (центр – 3)
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        neighborhoods = np.lib.stride_tricks.sliding_window_view(padded, (7, 7))
        neighborhoods = neighborhoods.reshape(-1, 49)  # 7x7=49 пікселів

        # Застосовуємо правило 3x3 до центру
        center_3x3 = neighborhoods[:, [16, 17, 18, 23, 24, 25, 30, 31, 32]]  # Центральні 3x3 пікселі
        matches_3x3 = np.all(center_3x3 == self.best_3x3_rule[:-1], axis=1)

        # Рахуємо кількість активованих пікселів у 7x7 (без центру)
        active_pixels = np.sum(neighborhoods[:, :24], axis=1)  # Усі, крім центральних 3x3

        # Визначаємо, чи потрапляє кількість у діапазон [lower, upper]
        in_range = (active_pixels >= lower) & (active_pixels <= upper)

        # Комбінуємо умови: правило 3x3 І діапазон 7x7
        combined = matches_3x3 & in_range
        new_image = image.copy()
        new_image.flat[combined] = self.best_3x3_rule[-1]  # Значення з правила 3x3
        return new_image

    def optimize_range(self, max_iter=50):
        best_f1 = -1
        best_lower, best_upper = 0, 0

        for lower in tqdm(range(0, 25)):  # Можливі значення lower (0-24)
            for upper in range(lower, 25):  # upper >= lower
                result = self.apply_range_rule(self.binary_image, lower, upper)
                current_f1 = f1_score(self.canny_flat, result.flatten())

                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_lower, best_upper = lower, upper

        return best_lower, best_upper, best_f1