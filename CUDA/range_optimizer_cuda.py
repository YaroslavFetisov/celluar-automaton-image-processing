import cupy as cp
from tqdm import tqdm

# CUDA ядро для застосування правила з діапазоном
apply_range_rule_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void applyRangeRule(const unsigned char* image, unsigned char* output, 
                       const unsigned char* rule_3x3, int width, int height, 
                       int lower, int upper) {
        int x = blockIdx.x * blockDim.x + threadIdx.x + 3; // Радіус 7x7 -> зсув 3
        int y = blockIdx.y * blockDim.y + threadIdx.y + 3;

        if (x >= width + 3 || y >= height + 3) return;

        int idx = (y - 3) * width + (x - 3);

        // Перевірка центрального 3x3
        bool match_3x3 = true;
        int center_indices[9] = {17, 18, 19, 24, 25, 26, 31, 32, 33}; // Центральні 3x3 у 7x7
        for (int i = 0; i < 9; i++) {
            int n_idx = (y + (center_indices[i] / 7 - 3)) * (width + 6) + (x + (center_indices[i] % 7 - 3));
            if (image[n_idx] != rule_3x3[i]) {
                match_3x3 = false;
                break;
            }
        }

        // Підрахунок активованих пікселів у 7x7 (без центральних 3x3)
        int active_pixels = 0;
        for (int dy = -3; dy <= 3; dy++) {
            for (int dx = -3; dx <= 3; dx++) {
                if (dy >= -1 && dy <= 1 && dx >= -1 && dx <= 1) continue; // Пропустити центр 3x3
                int n_idx = (y + dy) * (width + 6) + (x + dx);
                active_pixels += image[n_idx];
            }
        }

        // Перевірка діапазону
        bool in_range = (active_pixels >= lower) && (active_pixels <= upper);

        // Застосування правила
        if (match_3x3 && in_range) {
            output[idx] = rule_3x3[9]; // Останній елемент правила
        } else {
            output[idx] = image[(y * (width + 6) + x)];
        }
    }
    ''', 'applyRangeRule')

# CUDA ядро для обчислення F1 метрики
f1_score_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void calculateF1(const unsigned char* y_true, const unsigned char* y_pred, 
                     int* tp, int* fp, int* fn, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size) return;

        atomicAdd(&tp[0], y_true[idx] == 1 && y_pred[idx] == 1);
        atomicAdd(&fp[0], y_true[idx] == 0 && y_pred[idx] == 1);
        atomicAdd(&fn[0], y_true[idx] == 1 && y_pred[idx] == 0);
    }
    ''', 'calculateF1')


def cupy_f1_score(y_true, y_pred):
    tp = cp.zeros(1, dtype=cp.int32)
    fp = cp.zeros(1, dtype=cp.int32)
    fn = cp.zeros(1, dtype=cp.int32)

    size = y_true.size
    block_size = 256
    grid_size = (size + block_size - 1) // block_size

    f1_score_kernel((grid_size,), (block_size,), (y_true, y_pred, tp, fp, fn, size))

    precision = tp[0] / (tp[0] + fp[0] + 1e-7)
    recall = tp[0] / (tp[0] + fn[0] + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    return float(f1)


class RangeOptimizer:
    def __init__(self, image_processor, best_3x3_rule):
        self.image_processor = image_processor
        self.best_3x3_rule = best_3x3_rule
        self.original_image = cp.asarray(image_processor.original_image)
        self.binary_image = cp.asarray(image_processor.binary_image)
        self.canny_flat = cp.asarray(image_processor.canny_flat)

    def apply_range_rule(self, image, lower, upper):
        pad = 3  # Радіус 7x7
        image_gpu = cp.asarray(image, dtype=cp.uint8)
        rule_gpu = cp.asarray(self.best_3x3_rule, dtype=cp.uint8)
        padded = cp.pad(image_gpu, pad, mode='reflect')
        output_gpu = cp.zeros_like(image_gpu, dtype=cp.uint8)

        block_size = (16, 16)
        grid_size = ((image.shape[1] + block_size[0] - 1) // block_size[0],
                     (image.shape[0] + block_size[1] - 1) // block_size[1])

        apply_range_rule_kernel(grid_size, block_size,
                                (padded, output_gpu, rule_gpu, image.shape[1], image.shape[0],
                                 lower, upper))

        return output_gpu

    def optimize_range(self, max_iter=50):
        best_f1 = -1
        best_lower, best_upper = 0, 0

        for lower in tqdm(range(0, 25)):
            for upper in range(lower, 25):
                result = self.apply_range_rule(self.binary_image, lower, upper)
                current_f1 = cupy_f1_score(self.canny_flat, result.flatten())

                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_lower, best_upper = lower, upper

        return best_lower, best_upper, best_f1