import cv2
import numpy as np
from fractions import Fraction

org_img = 'origin.jpg'
img_read = cv2.imread(org_img)
print(img_read.shape)

print(img_read)

# greyscale the image
img_gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
H, W = img_gray.shape
print(img_gray)
print(img_gray.shape)

# reshape function (image matrix to vector)
img_vector = np.zeros((H * W, 1))
for i in range(H):
    for j in range(W):
        img_vector[j + i * W] = img_gray[i, j]
print(img_vector)

# Horizontal motion matrix
img_H_motion = np.zeros((H * W, H * W))
for i in range(H * W):
    for j in range(H * W):
        if (j - i == 0) or (j - i == 1) or (i - j == 1):
            img_H_motion[i, j] = 1
print(img_H_motion)

# multiplication
img_H_blurred_vector = np.zeros((H * W, 1))
img_H_blurred_vector = Fraction(1,3) * np.dot(img_H_motion, img_vector)

print(img_H_blurred_vector)

# convert back to matrix
img_blurred = np.zeros((H, W))
for i in range(H):
    for j in range(W):
        img_blurred[i, j] = img_H_blurred_vector[W * i + j, 0]

print(img_blurred)
cv2.imwrite('Horizontal_motion_blurred_image.jpg', img_blurred)


