from copy import copy, deepcopy
from fractions import Fraction
import numpy as np
import cv2
from numpy.linalg import inv

motion_matrix = np.matrix([[2, -2, 0, 4], [3, -3, 0, -1], [-1, 6, 5, -7], [-5, 1, 0, 6]], dtype=float)
result_matrix = np.matrix([[2], [-18], [-26], [7]], dtype=float)
print('motion_matrix = \n', motion_matrix)

H, W = motion_matrix.shape

U = deepcopy(motion_matrix)


def swap(u, r, p):  # check element U[i,i]
    temp = np.zeros((1, H), dtype=float)
    temp_p = np.zeros((1, H), dtype=float)
    counter = r + 1

    while True:
        if u[counter, r] == 0:
            counter += 1
            continue
        else:
            break
    # swap U[i] with U[counter]
    for a in range(H):
        temp[0, a] = u[r, a]
        temp_p[0, a] = p[r, a]

        u[r, a] = u[counter, a]
        p[r, a] = p[counter, a]

        u[counter, a] = temp[0, a]
        p[counter, a] = temp_p[0, a]
    return u


# Permutation matrix
P = np.eye(H, dtype=float)

# gauss elimination
for i in range(H):
    for j in range(i + 1, H):
        # make sure U[i,i] != 0 and swap row in this case
        if U[i, i] == 0:
            # swap row function here including P
            swap(U, i, P)
        if U[j, i] != 0:
            # do the subtraction
            fraction = float(U[j, i] / U[i, i])
            for k in range(H):
                # subtract the row
                U[j, k] = U[j, k] - fraction * U[i, k]
print('U matrix = \n', U)
print('P matrix = \n', P)
P_times_A = np.dot(P, motion_matrix)
print('PA = UL = \n', P_times_A)


# Sum function for L matrix
def sum_known(row, col, lower, upper):
    # define variable sum
    S = 0
    constant = col
    for t in range(col + 1):
        S += lower[row, t] * upper[t, constant]
    return S


# get matrix L through LU = PA
# build matrix L
# NON FORMULA METHOD
L_NON = np.eye(H, dtype=float)
for j in range(H):
    for i in range(j + 1, H):
        L_NON[i, j] = (P_times_A[i, j] - sum_known(i, j, L_NON, U)) / U[j, j]

print('L matrix (NON FORMULA METHOD) = \n', L_NON)

# DIRECT FORMULA METHOD
#  get matrix L through formula using the swapped A matrix
L_FORMULA = np.eye(H, dtype=float)

# gauss elimination without swapping and should get the same answer
U2 = deepcopy(P_times_A)


def get_l(upper, row):
    for r in range(row + 1, H):
        L_FORMULA[r, row] = upper[r, row] / upper[row, row]
    return L_FORMULA


for i in range(H):
    # Get L matrix here
    get_l(U2, i)
    for j in range(i + 1, H):
        # make sure U[i,i] != 0 and swap row in this case
        if U2[j, i] != 0:
            # do the subtraction
            fraction = float(U2[j, i] / U2[i, i])
            for k in range(H):
                # subtract the row
                U2[j, k] = U2[j, k] - fraction * U2[i, k]

print('U matrix using swapped PA = \n', U2)
print('L matrix (FORMULA METHOD) = \n', L_FORMULA)

# get swapped D matrix
print('result matrix = \n', result_matrix)
B_swapped = np.dot(P, result_matrix)
print('swapped result matrix B = \n', B_swapped)

# we know that L*D = B_swapped
D = np.zeros((H, 1), dtype=float)


def d_sum(row, lower):
    sum_d = 0.0
    for r in range(row):
        sum_d += lower[row, r] * D[r, 0]
    return sum_d


for i in range(H):
    D[i, 0] = B_swapped[i, 0] - d_sum(i, L_FORMULA)

print('matrix D = \n', D)

# get X matrix sloven
# using UX = D

X = np.zeros((H, 1), dtype=float)


def x_sum(row, upper):
    sum_x = 0.0
    for r in range(H - row - 1):
        sum_x += upper[row, H - 1 - r] * X[H - 1 - r, 0]
    return sum_x


for i in range(H - 1, -1, -1):
    X[i, 0] = (D[i, 0] - x_sum(i, U2)) / U2[i, i]

print('matrix X = \n', X)

# Horizontal motion deblur
# using inside built-in function
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

# get motion matrix A
motion_matrix = np.zeros((H * W, H * W), dtype=float)
for i in range(H * W):
    for j in range(H * W):
        if (j - i == 0) or (j - i == 1) or (i - j == 1):
            motion_matrix[i, j] = 1
print('motion_matrix = \n', motion_matrix)

X = np.dot(inv(motion_matrix), img_vector)

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

cv2.imwrite('Horizontal_motion_Deblurred_image_test.jpg', img_Deblurred)


# Focus Lost Deblur
# using inside built-in function
img = 'Focus_Lost_blurred_image.jpg'
blur_img = cv2.imread(img)
img_gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
print('origin_image = \n', img_gray)

H, W = img_gray.shape

img_vector = np.zeros((H * W, 1))
for i in range(H):
    for j in range(W):
        img_vector[j + i * W] = img_gray[i, j]
print('img vector = \n', img_vector)

# get motion matrix A
motion_matrix = np.zeros((H * W, H * W))
for i in range(H * W):
    for j in range(H * W):
        if i == j:
            motion_matrix[i, j] = 4
        elif (abs(j - i) == 1) or (abs(j - i)) == W:
            motion_matrix[i, j] = 1

print('motion matrix = \n',motion_matrix)

X = np.dot(inv(motion_matrix), img_vector)

# convert back to matrix
img_Deblurred = np.zeros((H, W), dtype=float)
for i in range(H):
    for j in range(W):
        img_Deblurred[i, j] = X[W * i + j, 0]
        if img_Deblurred[i, j] > 255:
            img_Deblurred[i, j] = 255
        elif img_Deblurred[i, j] < 0:
            img_Deblurred[i, j] = 0

print('recovered Focus Lost image = \n', img_Deblurred)

cv2.imwrite('Focus_Lost_Deblurred_image_test.jpg', img_Deblurred)
