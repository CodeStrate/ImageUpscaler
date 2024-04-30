import numpy as np
from scipy.signal import convolve2d

def unsharp_mask(image, kernel, amount=1.5):
    # Convert the image to float32
    image_float = image.astype(np.float32)

    # Initialize sharpened image array
    sharpened = np.zeros_like(image_float)

    # Apply unsharp mask to each channel separately
    for channel_idx in range(image_float.shape[2]):
        channel = image_float[:, :, channel_idx]
        
        # Apply convolution to the channel
        blurred_channel = convolve2d(channel, kernel, mode='same', boundary='symm')
        
        # Calculate sharpened channel
        sharpened_channel = np.clip(channel + amount * (channel - blurred_channel), 0, 255)
        
        # Store the sharpened channel in the output array
        sharpened[:, :, channel_idx] = sharpened_channel
    
    sharpened = sharpened.astype(np.uint8)
    return sharpened
