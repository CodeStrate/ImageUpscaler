import numpy as np
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
    

def lanczos_interpolation_2d(image, scale_factor, parallel=True):
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    output_image = Image.new("RGB", (new_width, new_height))

    def interpolate_row(row_index):
        row = []
        for j in range(new_width):
            x = j / scale_factor
            left = int(np.floor(x - 2))
            right = int(np.ceil(x + 2))

            interpolated_pixel = np.zeros(3)
            weight_sum = 0.0
            for k in range(left, right + 1):
                # Ensure that k is within the valid range of the input image
                if k >= 0 and k < width:
                    weight = lanczos_kernel(x - k)
                    pixel = np.array(image.getpixel((k, row_index))) * weight
                    interpolated_pixel += pixel
                    weight_sum += weight

            interpolated_pixel /= weight_sum if weight_sum != 0 else 1.0
            row.append(tuple(interpolated_pixel.astype(np.uint8)))

        return row

    if parallel:
        num_cores = min(8, new_height)  # Adjust the number of cores as needed
        rows = Parallel(n_jobs=num_cores)(delayed(interpolate_row)(i) for i in tqdm(range(new_height)))
        for y, row in enumerate(rows):
            output_image.putdata(row, y)
    else:
        for y in tqdm(range(new_height)):
            row = interpolate_row(y)
            output_image.putdata(row, y)

    return output_image


# Load the image
input_image = Image.open("examples\\pin.jpeg")

# Upscale the image using Lanczos interpolation with multiprocessing and progress bar
scaled_image = lanczos_interpolation_2d(input_image, 1.5)

# Save the scaled image
scaled_image.save("examples\\lanczos.png")
