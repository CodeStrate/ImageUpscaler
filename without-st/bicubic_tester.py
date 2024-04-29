import numpy as np
import math
from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image
from multiprocessing import Lock

# Define a lock for synchronization
lock = Lock()

# THE KERNEL FUNCTION h(x)
def interpolation_kernel(x, a):
    if (abs(x) >= 0) & (abs(x) <= 1):
        return (a + 2) * (abs(x)**3) - (a + 3) * (abs(x)**2) + 1
    elif (abs(x) > 1) & (abs(x) <= 2):
        return a * (abs(x)**3) - (5*a) * (abs(x)**2) + 8*a*(abs(x)) - 4*a
    else:
        return 0

def padding(image, H, W, C): 
    padded_image = np.zeros((H+4, W+4, C))
    padded_image[2:H+2, 2:W+2, :C] = image

    # pad the first and last two rows and cols
    padded_image[2:H+2, 0:2, :C] = image[:, 0:1, :C]
    padded_image[H+2:H+4, 2:W+2, :] = image[H-1:H, :, :]
    padded_image[2:H+2, W+2:W+4, :] = image[:, W-1:W, :]
    padded_image[0:2, 2:W+2, :C] = image[0:1, :, :C]

    # pad the final eight points remaining
    padded_image[0:2, 0:2, :C] = image[0, 0, :C]
    padded_image[H+2:H+4, 0:2, :C] = image[H-1, 0, :C]
    padded_image[H+2:H+4, W+2:W+4, :C] = image[H-1, W-1, :C]
    padded_image[0:2, W+2:W+4, :C] = image[0, W-1, :C]

    return padded_image

def bicubic_interpolation(image, upscale_ratio, a):
    # image size
    H, W, C = image.shape

    image = padding(image, H, W, C)

    # create new image
    dH = math.floor(H*upscale_ratio)
    dW = math.floor(W*upscale_ratio)
    dest_image = np.zeros((dH, dW, 3))

    h = 1/upscale_ratio

    print('Start Upscaling using Bicubic...')
    print('IT WILL TAKE TIME! So be patient... ^^')

    def process_pixel(c, j, i):
        x, y = i*h+2, j*h+2

        x1 = 1 + x - math.floor(x)
        x2 = x - math.floor(x)
        x3 = math.floor(x) + 1 - x
        x4 = math.floor(x) + 2 - x

        y1 = 1 + y - math.floor(y)
        y2 = y - math.floor(y)
        y3 = math.floor(y) + 1 - y
        y4 = math.floor(y) + 2 - y

        matrix_l = np.matrix([[interpolation_kernel(x1, a),
                               interpolation_kernel(x2, a),
                               interpolation_kernel(x3, a),
                               interpolation_kernel(x4, a)]])

        matrix_m = np.matrix([[image[int(y-y1), int(x-x1), c],
                               image[int(y-y2), int(x-x1), c],
                               image[int(y+y3), int(x-x1), c],
                               image[int(y+y4), int(x-x1), c]],
                              [image[int(y-y1), int(x-x2), c],
                               image[int(y-y2), int(x-x2), c],
                               image[int(y+y3), int(x-x2), c],
                               image[int(y+y4), int(x-x2), c]],
                              [image[int(y-y1), int(x+x3), c],
                               image[int(y-y2), int(x+x3), c],
                               image[int(y+y3), int(x+x3), c],
                               image[int(y+y4), int(x+x3), c]],
                              [image[int(y-y1), int(x+x4), c],
                               image[int(y-y2), int(x+x4), c],
                               image[int(y+y3), int(x+x4), c],
                               image[int(y+y4), int(x+x4), c]]])

        matrix_r = np.matrix([[interpolation_kernel(y1, a)],
                               [interpolation_kernel(y2, a)],
                               [interpolation_kernel(y3, a)],
                               [interpolation_kernel(y4, a)]])

        result = np.dot(np.dot(matrix_l, matrix_m), matrix_r)

        return result

    # Using joblib for parallel processing with lock
    results = Parallel(n_jobs=-1)(delayed(process_pixel)(c, j, i) for c in range(C) for j in tqdm(range(dH)) for i in range(dW))

    progress = 0
    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                with lock:
                    dest_image[j, i, c] = results[progress][0]  # Access the first element of the result array
                progress += 1

    return dest_image
# Example usage
input_image = Image.open('examples\\seagull.jpg')

image = np.array(input_image)

# Define upscale ratio and interpolation parameter 'a'
upscale_ratio = 2
a = -0.5

# Perform bicubic interpolation
result_image = bicubic_interpolation(image, upscale_ratio, a)
result_image_pil = Image.fromarray(result_image.astype(np.uint8))
result_image_pil.save('test.png')
