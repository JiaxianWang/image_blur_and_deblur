import cv2
import numpy as np

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

# focus lost blur matrix
img_FL_motion = np.zeros((H * W, H * W))
for i in range(H * W):
    for j in range(H * W):
        if i == j:
            img_FL_motion[i, j] = 4
        elif (abs(j - i) == 1) or (abs(j - i)) == W:
            img_FL_motion[i, j] = 1

print(img_FL_motion)

# multiplication
img_FL_blurred_vector = np.zeros((H * W, 1))
img_FL_blurred_vector = 1 / 8 * np.dot(img_FL_motion, img_vector)

# convert back to matrix
img_blurred = np.zeros((H, W))
for i in range(H):
    for j in range(W):
        img_blurred[i, j] = img_FL_blurred_vector[W * i + j, 0]

cv2.imwrite('Focus_lost_blurred_image.jpg', img_blurred)
