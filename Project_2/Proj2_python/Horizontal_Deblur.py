from fractions import Fraction
import cv2
import numpy as np
from copy import copy, deepcopy

img = 'Horizontal_motion_blurred_image.jpg'
blur_img = cv2.imread(img)
img_gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
print('origin_image = \n', img_gray)

H, W = img_gray.shape

img_vector = np.zeros((H * W, 1))
for i in range(H):
    for j in range(W):
        img_vector[j + i * W] = img_gray[i, j]
print('img vector = \n', img_vector)


def swap(u, r, p):  # check element U[i,i]
    temp = np.zeros((1, H * W), dtype=float)
    temp_p = np.zeros((1, H * W), dtype=float)
    counter = r + 1

    while True:
        if counter == H*W:
            break
        if u[counter, r] == 0:
            counter += 1
            continue
        else:
            break
    # swap U[i] with U[counter]
    for a in range(H * W):
        temp[0, a] = u[r, a]
        temp_p[0, a] = p[r, a]

        u[r, a] = u[counter, a]
        p[r, a] = p[counter, a]

        u[counter, a] = temp[0, a]
        p[counter, a] = temp_p[0, a]
    return u


# LU decomposition

# starting from gauss elimination for A to get U

# with permutation matrix P indicated the row swapping
P = np.eye(H * W, dtype=float)

# get motion matrix A
motion_matrix = np.zeros((H * W, H * W), dtype=float)
for i in range(H * W):
    for j in range(H * W):
        if (j - i == 0) or (j - i == 1) or (i - j == 1):
            motion_matrix[i, j] = 1
print('motion_matrix = \n', motion_matrix)

# gauss elimination in A to get U
U = deepcopy(motion_matrix)
for i in range(H * W):
    for j in range(i + 1, H * W):
        # make sure U[i,i] != 0 and swap row in this case
        if U[i, i] == 0:
            # swap row function here with permutation matrix
            swap(U, i, P)
        if U[j, i] != 0:
            # do the subtraction
            fraction = float(U[j, i] / U[i, i])
            for k in range(H * W):
                # subtract the row
                U[j, k] = float(U[j, k]) - fraction * float(U[i, k])

print('U matrix = \n', U)
print('P matrix = \n', P)
P_times_A = np.dot(P, motion_matrix)
print('PA = UL = \n', P_times_A)

# get matrix L through LU = PA
# build matrix L
# FORMULA METHOD
#  get matrix L through formula using the swapped A matrix
L_FORMULA = np.eye(H * W, dtype=float)

# gauss elimination without swapping and should get the same answer
U2 = deepcopy(P_times_A)


def get_l(upper, row):
    for r in range(row + 1, H * W):
        L_FORMULA[r, row] = upper[r, row] / upper[row, row]
    return L_FORMULA


for i in range(H * W):
    # Get L matrix here
    get_l(U2, i)
    for j in range(i + 1, H * W):
        # make sure U[i,i] != 0 and swap row in this case
        if U2[j, i] != 0:
            # do the subtraction
            fraction = float(U2[j, i] / U2[i, i])
            for k in range(H * W):
                # subtract the row
                U2[j, k] = U2[j, k] - fraction * U2[i, k]

print('U matrix using swapped PA = \n', U2)
print('L matrix (FORMULA METHOD) = \n', L_FORMULA)

# get swapped D matrix

print('result matrix = \n', img_vector)
B_swapped = np.dot(P, img_vector)
print('swapped result matrix B = \n', B_swapped)

# we know that L*D = B_swapped
D = np.zeros((H * W, 1), dtype=float)


def d_sum(row, lower):
    sum_d = 0.0
    for r in range(row):
        sum_d += lower[row, r] * D[r, 0]
    return sum_d


for i in range(H * W):
    D[i, 0] = B_swapped[i, 0] - d_sum(i, L_FORMULA)

print('matrix D = \n', D)

# get X matrix sloven
# using UX = D

X = np.zeros((H * W, 1), dtype=float)


def x_sum(row, upper):
    sum_x = 0.0
    for r in range(H * W - row - 1):
        sum_x += upper[row, H * W - 1 - r] * X[H * W - 1 - r, 0]
    return sum_x


for i in range(H * W - 1, -1, -1):
    X[i, 0] = (D[i, 0] - x_sum(i, U2)) / U2[i, i]

print('matrix X = \n', X)

# convert back to matrix
img_Deblurred = np.zeros((H, W), dtype=float)
for i in range(H):
    for j in range(W):
        img_Deblurred[i, j] = X[W * i + j, 0]
        if img_Deblurred[i, j] > 255:
            img_Deblurred[i, j] = 255
        elif img_Deblurred[i, j] < 0:
            img_Deblurred[i, j] = 0

print('recovered Horizontal motion image = \n', img_Deblurred)

cv2.imwrite('Horizontal_motion_Deblurred_image.jpg', img_Deblurred)
