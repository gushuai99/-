import math
import numpy as np
import cv2
import glob
import os


def EME(rbg, L):
    m, n = np.shape(rbg)  # 横向为n列 纵向为m行
    number_m = math.floor(m / L)
    number_n = math.floor(n / L)
    # A1 = np.zeros((L, L))
    m1 = 0
    E = 0
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            A1 = rbg[m1:m1 + L, n1:n1 + L]
            rbg_min = np.amin(np.amin(A1))
            rbg_max = np.amax(np.amax(A1))

            if rbg_min > 0:
                rbg_ratio = rbg_max / rbg_min
            else:
                rbg_ratio = rbg_max
            E = E + np.log(rbg_ratio + 1e-5)

            n1 = n1 + L
        m1 = m1 + L
    E_sum = 2 * E / (number_m * number_n)
    return E_sum


#计算灰度图的EME值,k、l分别是单元格的横纵像素数
def cal_EME(image, k=5, l=5):
    number = image.size  # 计算 X 中所有元素的个数
    image_row = np.size(image, 0)  # 计算 X 的行数
    image_col = np.size(image, 1)  # 计算 X 的列数
    eme = []
    #引入修正因子q
    q = 0.0001
    k1 = math.floor(image_row / k)
    k2 = math.floor(image_col / l)
    for i in range(k1):
        for j in range(k2):
            single_matrix = image[i * k:((i + 1) * k), j * l:((j + 1) * l)]
            a = np.max(single_matrix)
            b = np.min(single_matrix) + q
            if a > 0:
                single_eme = 20 * math.log(np.max(single_matrix) / (np.min(single_matrix) + q))
                eme.append(single_eme)
            else:
                pass
    cal_EME = np.abs(np.sum(eme) / (k1 * k2))
    return cal_EME


if __name__ == '__main__':
    testimage_pathlist = glob.glob(r".\contrast\*.jpg")
    testimage_pathlist.sort()
    for i in range(len(testimage_pathlist)):
        img = cv2.imread(testimage_pathlist[i], 0)
        eme = cal_EME(img, 4)
        fileName = os.path.basename(testimage_pathlist[i])
        print(f'{fileName}的eme值为{eme}')
