# LINEAR INTERPOLATION :
- A technique used in forecasting, we find unknown values using the set of given values

## Formula : y = y1 + (x - x1) * (y2 - y1)/(x2 - x1)
where x1, y1 = first point
x2, y2 = second point
x = point to interpolate
y is the interpolated value

# Example

- Find y if x = 6 and points given are (3,4) and (6, 8)

using formula y = 4 + (6-3) * (8-4)/(6-3) = 4 + 3*(4/3) = 4 + 4 = 8.


# BICUBIC INTERPOLATION
- In addition to going 2×2 neighborhood of known pixel values, Bicubic goes one step beyond bilinear by considering the closest 4×4 neighborhood of known pixels — for a complete of 16 pixels.
- The pixels that are closer to the one that’s to be estimated are given higher weights as compared to those that are further away.
- Therefore, the farthest pixels have the smallest amount of weight.
- The results of Bicubic interpolation are far better as compared to NN or bilinear algorithms.
- This can be because a greater number of known pixel values are considered while estimating the desired value.

- Thus, making it one of all the foremost standard interpolation methods.

# MY IMPLEMENTATION FOR BICUBIC INTERPOLATION FOR IMAGES.

- h(x) = (a+2)|x|^3 - (a + 3)|x|^2 + 1 if |x| b/w 0 and 1 , 1 being exclusive
    - h(x) = a|x|^3 - 5a|x|^2 + 8a|x| - 4a if |x| b/w 1 and 2 , 2 being exclusive
    - h(x) = 0 if x > 2 

    - a is a coefficient with value range(-0.5 to -0.75)