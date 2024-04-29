# mask = original_image + amount * (original - blurred)

import numpy as np

def unsharp_mask(image, kernel, kernel_size=(5, 5), sigma=1.0, amount=1.5):
    # Convert the image to float32
    image_float = image.astype(np.float32)

    # Separate channels
    channels = []
    for i in range(image_float.shape[2]):
        channel = image_float[:, :, i]
        channels.append(channel)

    # Apply unsharp mask to each channel
    sharpened_channels = []
    for channel in channels:
        blurred = np.zeros_like(channel)
        for i in range(channel.shape[0] - kernel_size[0] + 1):
            for j in range(channel.shape[1] - kernel_size[1] + 1):
                blurred[i, j] = np.sum(channel[i:i+kernel_size[0], j:j+kernel_size[1]] * kernel)
        sharpened_channel = np.clip(channel + amount * (channel - blurred), 0, 255)
        sharpened_channels.append(sharpened_channel)

    # Merge sharpened channels
    sharpened = np.stack(sharpened_channels, axis=2).astype(np.uint8)
    
    return sharpened