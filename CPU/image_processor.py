import cv2
import numpy as np
from sklearn.metrics import f1_score

class ImageProcessor:
    def __init__(self, image_path, neighborhood_size):
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, self.binary_image = cv2.threshold(self.original_image, 128, 1, cv2.THRESH_BINARY)
        self.canny_edges = cv2.Canny(self.original_image, 100, 200)
        self.canny_edges = (self.canny_edges > 0).astype(np.uint8)
        self.canny_flat = self.canny_edges.flatten()
        self.neighborhood = neighborhood_size

    def apply_rule(self, image, rule):
        pad = self.neighborhood // 2
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        neighborhoods = np.lib.stride_tricks.sliding_window_view(padded, (self.neighborhood, self.neighborhood))
        neighborhoods = neighborhoods.reshape(-1, self.neighborhood ** 2)
        matches = np.all(neighborhoods == rule[:-1], axis=1)
        new_image = image.copy()
        new_image.flat[matches] = rule[-1]
        return new_image

    def evaluate_rule(self, rule):
        result = self.apply_rule(self.binary_image, rule)
        return f1_score(self.canny_flat, result.flatten())
