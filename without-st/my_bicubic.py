import cv2, time, sys, math, numpy as np

# MY IMPLEMENTATION FOR BICUBIC INTERPOLATION FOR IMAGES.

# h(x) = (a+2)|x|^3 - (a + 3)|x|^2 + 1 if |x| b/w 0 and 1 , 1 being exclusive
# h(x) = a|x|^3 - 5a|x|^2 + 8a|x| - 4a if |x| b/w 1 and 2 , 2 being exclusive
# h(x) = 0 if x > 2 

# a is a coefficient with value range(-0.5 to -0.75)

# THE KERNEL FUNCTION h(x)
def interpolation_kernel(x, a):
    if (abs(x) >= 0) & (abs(x) <= 1):
        return (a + 2) * (abs(x)**3) - (a + 3) * (abs(x)**2) + 1
    elif (abs(x) > 1) & (abs(x) <= 2):
        return a * (abs(x)**3) - (5*a) * (abs(x)**2) + 8*a*(abs(x)) - 4*a
    else:
        return 0


# define padding to add borders to the image such that it can compute the pixel values along said border/boundary

def padding(image, H, W, C): # image is the input image , H is height, W is width and C is color channels.

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


def progress_bar(progress):

    MAX_LENGTH = 30
    BAR_LENGTH = int(MAX_LENGTH * progress)

    return ('Progress : [' + '=' * BAR_LENGTH + ('>' if BAR_LENGTH < MAX_LENGTH else '')
            + ' ' * (MAX_LENGTH - BAR_LENGTH) + '] %.1f%%' % (progress * 100.))

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

    progress = 0

    for c in range(C):
        for j in range(dH):
            for i in range(dW):

                x, y = i*h+2, j*h+2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                # considering all nearby 16 values

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
                
                # dot producting all 3 matrices

                dest_image[j, i, c] = np.dot(np.dot(matrix_l, matrix_m), matrix_r)

                progress = progress + 1
                sys.stderr.write('\r\033[K' + progress_bar(progress/(C*dH*dW)))
                sys.stderr.flush()
    sys.stderr.write('\n')
    sys.stderr.flush()

    return dest_image


if __name__ == '__main__':
    image = cv2.imread('examples\\butterfly.png')
    ratio = 2
    a = -1/2

    dst = bicubic_interpolation(image, ratio, a)
    print('Completed!')

    cv2.imwrite('examples\\butterfly_bicubic.png', dst)

    bicubic_image = cv2.imread('examples\\butterfly_bicubic.png')

    print('Original Image shape : ', image.shape)
    print('Bicubic Generated Image shape : ', bicubic_image.shape)