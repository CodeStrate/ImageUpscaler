import numpy as np
import math
from joblib import Parallel, delayed
from multiprocessing import Lock
from tqdm import tqdm

lock = Lock()

def lanczos_kernel(x, a=3):
    if x == 0:
        return 1
    if -a <= x < a:
        return (a * np.sin(np.pi * x)) * np.sin((np.pi * x) / a) / (np.pi * np.pi * x * x)
    else:
        return 0

def lanczos_interpolation(image, upscale_ratio):
    H, W, C = image.shape
    dH = math.floor(H * upscale_ratio)
    dW = math.floor(W * upscale_ratio)
    dest_image = np.zeros((dH, dW, 3))
    h = 1 / upscale_ratio

    def process_pixel(j, i):
        x, y = i * h + 2, j * h + 2
        x1 = math.floor(x)
        y1 = math.floor(y)
        interpolated_values = []
        for c in range(C):
            sum_val = 0
            weight_sum = 0
            for dx in range(-2, 2 + 1):
                for dy in range(-2, 2 + 1):
                    px = x1 + dx
                    py = y1 + dy
                    if px >= 0 and px < W and py >= 0 and py < H:
                        wx = lanczos_kernel(x - px)
                        wy = lanczos_kernel(y - py)
                        weight = wx * wy
                        weight_sum += weight
                        sum_val += weight * image[py, px, c]
            interpolated_value = sum_val / weight_sum if weight_sum != 0 else 0
            interpolated_values.append(interpolated_value)
        return interpolated_values

    total_pixels = dH * dW
    results = Parallel(n_jobs=-1)(delayed(process_pixel)(j, i) for j in range(dH) for i in range(dW))

    progress_bar = tqdm(total=total_pixels, desc="Interpolating", unit="pixel")
    for j in range(dH):
        for i in range(dW):
            with lock:
                dest_image[j, i] = results[j * dW + i]
                progress_bar.update(1)
    progress_bar.close()

    return dest_image
