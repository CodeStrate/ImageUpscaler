import numpy as np
from scipy.ndimage import convolve
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm

def lanczos_kernel(x, a=3):
    if x == 0:
        return 1
    if -a <= x < a:
        return (a * np.sin(np.pi * x)) * np.sin((np.pi * x) / a) / (np.pi * np.pi * x * x)
    else:
        return 0
    
def lanczos_interpolation(image, upscale_ratio, a=3):
    # Define the Lanczos kernel size based on 'a'
    kernel_size = 2 * a

    # Upscale the image dimensions
    new_height = int(image.shape[0] * upscale_ratio)
    new_width = int(image.shape[1] * upscale_ratio)

    # Generate Lanczos kernel weights
    weights_x = np.array([lanczos_kernel((i + 0.5) / upscale_ratio - 1.5, a) for i in range(kernel_size)])
    weights_y = np.array([lanczos_kernel((i + 0.5) / upscale_ratio - 1.5, a) for i in range(kernel_size)])

    # Normalize weights
    weights_x /= weights_x.sum()
    weights_y /= weights_y.sum()

    def process_row(row):
        return convolve(row, weights_x[np.newaxis, :], mode='mirror')

    def process_image_slice(slice):
        return np.array([process_row(row) for row in slice])

    # Use joblib and tqdm for parallel processing with progress bar
    progress_bar = tqdm(total=new_height, desc="Interpolating", unit="row")
    results = Parallel(n_jobs=-1)(delayed(process_image_slice)(image[i:i+kernel_size])
                                   for i in range(0, new_height, kernel_size))
    progress_bar.close()

    # Stack the results along a new axis after reshaping them to have the same number of dimensions
    max_dim = max(len(result.shape) for result in results)
    results = [np.expand_dims(result, axis=-1) for result in results if len(result.shape) < max_dim]
    interpolated_image = np.concatenate(results, axis=-1)

    # Apply Lanczos interpolation along y-axis
    interpolated_image = convolve(interpolated_image, weights_y[:, np.newaxis], mode='mirror')

    # Crop the interpolated image to match the target size
    crop_start_h = max((interpolated_image.shape[0] - new_height) // 2, 0)
    crop_start_w = max((interpolated_image.shape[1] - new_width) // 2, 0)
    crop_end_h = min(crop_start_h + new_height, interpolated_image.shape[0])
    crop_end_w = min(crop_start_w + new_width, interpolated_image.shape[1])
    interpolated_image = interpolated_image[crop_start_h:crop_end_h, crop_start_w:crop_end_w]

    return interpolated_image.astype(np.uint8)

# Load the input image
input_image = Image.open('examples\\bf_alt.png')
image = np.array(input_image)

# Define upscale ratio
upscale_ratio = 2

# Perform Lanczos interpolation
result_image = lanczos_interpolation(image, upscale_ratio)

# Convert the interpolated image array back to PIL Image format
result_image_pil = Image.fromarray(result_image.astype(np.uint8))

# Display the interpolated image
result_image_pil.show()
