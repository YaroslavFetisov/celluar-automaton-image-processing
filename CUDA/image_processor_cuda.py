import cv2
import cupy as cp

# CUDA ядро для застосування правила
apply_rule_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void applyRule(const unsigned char* image, unsigned char* output, 
                   const unsigned char* rule, int width, int height, 
                   int neighborhood_size, int rule_size) {
        int x = blockIdx.x * blockDim.x + threadIdx.x + neighborhood_size / 2;
        int y = blockIdx.y * blockDim.y + threadIdx.y + neighborhood_size / 2;

        if (x >= width + neighborhood_size / 2 || y >= height + neighborhood_size / 2) return;

        int idx = (y - neighborhood_size / 2) * width + (x - neighborhood_size / 2);
        int pad = neighborhood_size / 2;
        bool match = true;

        for (int dy = -pad; dy <= pad; dy++) {
            for (int dx = -pad; dx <= pad; dx++) {
                int n_idx = (y + dy) * (width + 2 * pad) + (x + dx);
                int rule_idx = (dy + pad) * neighborhood_size + (dx + pad);
                if (image[n_idx] != rule[rule_idx]) {
                    match = false;
                    break;
                }
            }
            if (!match) break;
        }

        if (match) {
            output[idx] = rule[rule_size - 1];
        } else {
            output[idx] = image[(y * (width + 2 * pad) + x)];
        }
    }
    ''', 'applyRule')

class ImageProcessor:
    def __init__(self, image_path, neighborhood_size):
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, self.binary_image = cv2.threshold(self.original_image, 128, 1, cv2.THRESH_BINARY)
        self.canny_edges = cv2.Canny(self.original_image, 100, 200)
        self.canny_edges = (self.canny_edges > 0).astype('uint8')
        self.canny_flat = self.canny_edges.flatten()
        self.neighborhood = neighborhood_size

    def apply_rule(self, image, rule):
        pad = self.neighborhood // 2
        image_gpu = cp.asarray(image, dtype=cp.uint8)
        rule_gpu = cp.asarray(rule, dtype=cp.uint8)
        padded = cp.pad(image_gpu, pad, mode='reflect')
        output_gpu = cp.zeros_like(image_gpu, dtype=cp.uint8)

        block_size = (16, 16)
        grid_size = ((image.shape[1] + block_size[0] - 1) // block_size[0],
                     (image.shape[0] + block_size[1] - 1) // block_size[1])

        apply_rule_kernel(grid_size, block_size,
                         (padded, output_gpu, rule_gpu, image.shape[1], image.shape[0],
                          self.neighborhood, self.neighborhood ** 2))

        return output_gpu