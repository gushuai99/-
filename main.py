import os
import cv2
import natsort
import numpy as np
from sceneRadianceCLAHE import RecoverCLAHE

#CLAHE
np.seterr(over='ignore')
if __name__ == '__main__':
    pass

folder = (r'C:\Users\gushuai\Desktop\Single-Underwater-Image-Enhancement-and-Color-Restoration-master\Underwater Image '
          r'Enhancement\RGHS')
path = folder + "/InputImages"
files = os.listdir(path)
files = natsort.natsorted(files)
for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.j')[0]
    if os.path.isfile(filepath):
        print('********    file   ********\n', file)
        # img = cv2.imread('InputImages/' + file)
        img = cv2.imread(folder + '/InputImages/' + file)
        sceneRadiance_CLHAE = RecoverCLAHE(img)
        # cv2.imwrite('OutputImages/' + prefix + '_CLAHE.jpg', sceneRadiance)
    cv2.imshow("CLHAE", sceneRadiance_CLHAE)
    #cv2.waitKey(0)

    #对边零填充
    top, bottom, left, right = 2, 1, 0, 1
    sceneRadiance_CLHAE00 = cv2.copyMakeBorder(sceneRadiance_CLHAE, top, bottom, left, right,
                                               borderType=cv2.BORDER_CONSTANT,
                                               value=0)  #(336,476,3)
    #print(sceneRadiance_CLHAE00.shape)

    #高斯卷积
    # 生成一个大小为 5，标准差为 1 的高斯核
    '''
    ksize = 5
    sigma = 1
    gaussian_kernel = cv2.getGaussianKernel(ksize, sigma)
    # 使用高斯核进行卷积操作
    #print(sceneRadiance_CLHAE.shape)
    #b = np.pad(sceneRadiance_CLHAE, 1, 'constant')
    blurred_CLAHE1 = cv2.filter2D(sceneRadiance_CLHAE00, -1, gaussian_kernel)  #(335,474,3)
    '''
    #下采样
    upscaled_CLAHE1 = cv2.pyrDown(sceneRadiance_CLHAE00)  #(168,238,3)
    # print(upscaled_CLAHE1.shape)

    #blurred_CLAHE2 = cv2.filter2D(upscaled_CLAHE1, -1, gaussian_kernel)  #(168,238,3)
    #  print(blurred_CLAHE2.shape)
    upscaled_CLAHE2 = cv2.pyrDown(upscaled_CLAHE1)  #(84,119,3)

    #一次上采样卷积
    upscaled_CLAHE_11 = cv2.pyrUp(upscaled_CLAHE1)  #(336,476,3)
    #blurred_CLAHE_11 = cv2.filter2D(upscaled_CLAHE_11, -1, gaussian_kernel)
    #print(blurred_CLAHE_11.shape)

    #二次上采样卷积
    upscaled_CLAHE_22 = cv2.pyrUp(upscaled_CLAHE2)
    #blurred_CLAHE_22 = cv2.filter2D(upscaled_CLAHE_22, -1, gaussian_kernel)

#输出图像尺寸来验证是否相同
#sp1=blurred_CLAHE.shape
#sp2=sceneRadiance_CLHAE.shape
#print(sp1)
#print(sp2)
#输出图像矩阵来验证是否相同
#a = np.array(blurred_CLAHE_11)
#b = np.array(sceneRadiance_CLHAE)
#print(b)
#print("---------------other----------------")
#print(a)

#不相同则调整图像大小
#new_size = (335, 474, 3)
#resized_image = np.resize(blurred_CLAHE_11, new_size)

#new_size_2 = (168, 238, 3)
#resized_image_2 = np.resize(upscaled_CLAHE1, new_size_2)
#print(sceneRadiance_CLHAE00.shape)
#print(blurred_CLAHE_11.shape)
#两图像相减
#一次
#imge_1 = abs(blurred_CLAHE_11 - resized_image)

#top,bottom, left, right = 0, 0, 1,0
#constant_3 = cv2.copyMakeBorder(upscaled_CLAHE1, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
#value=0)

imge_1 = cv2.subtract(upscaled_CLAHE_11, sceneRadiance_CLHAE00)  #(336,476,3)

#print(imge_1.shape)
#二次
#print(upscaled_CLAHE1.shape)
#print(blurred_CLAHE_22.shape)
#imge_2 = abs(resized_image_2- blurred_CLAHE_22)
imge_2 = cv2.subtract(upscaled_CLAHE1, upscaled_CLAHE_22)

'''
cv2.imshow("11", imge_2)
cv2.waitKey(0)
'''

#UCM
from color_equalisation import RGB_equalisation
from global_histogram_stretching import stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/NonPhysical/UCM"
folder = (r'C:\Users\gushuai\Desktop\Single-Underwater-Image-Enhancement-and-Color-Restoration-master\Underwater Image '
          r'Enhancement\RGHS')

path = folder + '/InputImages'
files = os.listdir(path)
files = natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********\n', file)
        # img = cv2.imread('InputImages/' + file)
        img = cv2.imread(folder + '/InputImages/' + file)
        # print('Number',Number)
        sceneRadiance = RGB_equalisation(img)
        sceneRadiance = stretching(sceneRadiance)
        #cv2.imwrite(folder + '/OutputImages/' + Number + 'Stretched.jpg', sceneRadiance)
        sceneRadiance = HSVStretching(sceneRadiance)
        sceneRadiance_UCM = sceneRadianceRGB(sceneRadiance)  #(335,474,3)

        top, bottom, left, right = 2, 1, 0, 1
        sceneRadiance_UCM00 = cv2.copyMakeBorder(sceneRadiance_UCM, top, bottom, left, right,
                                                 borderType=cv2.BORDER_CONSTANT,
                                                 value=0)  # (336,476,3)

    cv2.imshow("sceneRadiance_UCM", sceneRadiance_UCM)
    #cv2.waitKey(0)
    #endtime = datetime.datetime.now()
    #time = endtime-starttime
    #print('time',time)
    #高斯卷积
    # 使用高斯核进行卷积操作
    #blurred_UCM1 = cv2.filter2D(sceneRadiance_UCM00, -1, gaussian_kernel)
    #下采样
    upscaled_UCM1 = cv2.pyrDown(sceneRadiance_UCM00)

    #blurred_UCM2 = cv2.filter2D(upscaled_UCM1, -1, gaussian_kernel)
    upscaled_UCM2 = cv2.pyrDown(upscaled_UCM1)

    #一次上采样
    upscaled_UCM_11 = cv2.pyrUp(upscaled_UCM1)
    # blurred_UCM_11 = cv2.filter2D(upscaled_UCM_11, -1, gaussian_kernel)

    #第二次上采样
    upscaled_UCM_22 = cv2.pyrUp(upscaled_UCM2)
    # blurred_UCM_22 = cv2.filter2D(upscaled_UCM_22, -1, gaussian_kernel)

#new_size_01 = (335, 474, 3)
#resized_image_01 = np.resize(blurred_UCM_11, new_size_01)
#imge_01 = abs(resized_image_01 - blurred_UCM_11)
imge_01 = cv2.subtract(upscaled_UCM_11, sceneRadiance_UCM00)

#new_size_02 = (168, 238, 3)
#resized_image_02 = np.resize(sceneRadiance_UCM, new_size_02)
#imge_02 = abs(resized_image_02 - blurred_UCM_22)
imge_02 = cv2.subtract(upscaled_UCM1, upscaled_UCM_22)
#cv2.imshow(prefix, imge_02)
#cv2.waitKey(0)

# 图像融
'''
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


img = img.astype(np.float32) / 255.0
img_low_contrast = np.power(img, 1/2.2) * 255.0

# 将图像转换为uint8
img_low_contrast = np.uint8(img_low_contrast)
# 进行直方图均衡化
img_he = cv2.equalizeHist(img_low_contrast)
# 进行自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_low_contrast)

#高斯滤波
Gf=cv2.GaussianBlur(img,(5,5),0,0)

#高斯下采样
imag1 = cv2.pyrDown(Gf)

# 这里图像的尺寸必须为2的n次幂
A = clahe.apply(img_low_contrast)
A = cv2.resize(A, (512, 512), interpolation=cv2.INTER_CUBIC)
B = clahe.apply(img_low_contrast)
B = cv2.resize(B, (512, 512), interpolation=cv2.INTER_CUBIC)
# 生成8层的高斯金字塔gpA `
G = A.copy()
gpA = [G]
for i in range(7):
    # 进行7次高斯模糊+下采样
    G = cv2.pyrDown(G)
    # 把每次高斯模糊+下采样的结果送给gpA
    gpA.append(G)

# 生成8层的高斯金字塔gpB
G = B.copy()
gpB = [G]
for i in range(7):
    # 进行7次高斯模糊+下采样
    G = cv2.pyrDown(G)
    # 把每次高斯模糊+下采样的结果送给gpB
    gpB.append(G)

# 把两个高斯金字塔进行合并
LR = []
# zip(lpA,lpB)把两个高斯金字塔各层的两个图像组合成一个元组，然后各元组构成一个大zip
# 对于各元组中的两个图像
for la, lb in zip(gpA, gpB):
    # 取la或lb的尺寸皆可
    rows, dpt = la.shape
    # 利用np.hstack将这两个图像“一半一半”地拼接起来
    # 取la的左边一半和lb的右边一半拼成一个融合后的图，结果赋给ls
    lr = np.hstack((la[:, 0:dpt // 2], lb[:, dpt // 2:]))
    # 两个拉普拉斯金字塔各层图像融合后的结果赋给LS
    LR.append(lr)

# 用融合后的拉普拉斯金字塔重构出最终图像
# 初始化ls为融合后拉普拉斯金字塔的最高层
# 下面的循环结束后ls就是要求的最终结果图像
lr = LR[7]
for i in range(6, -1, -1):
    # 每层图像先上采样，再和当前层的下一层图像相加，结果再赋给ls
    lr = cv2.pyrUp(lr)
    lr = cv2.add(lr, LR[i])
# 生成8层拉普拉斯金字塔
# 从顶层开始构建
# 顶层即高斯金字塔的顶层
lpA = [gpA[7]]
# 7 6 5 4 3 2 1
for i in range(7, 0, -1):
    # 从顶层开始，不断上采样
    GE = cv2.pyrUp(gpA[i])
    # 用下一层的高斯减去上层高斯的上采样
    L = cv2.subtract(gpA[i - 1], GE)
    # 结果送给拉普拉斯金字塔
    lpA.append(L)

lpB = [gpB[7]]
for i in range(7, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i - 1], GE)
    lpB.append(L)

# 把两个拉普拉斯金字塔进行合并
LS = []
# zip(lpA,lpB)把两个拉普拉斯金字塔各层的两个图像组合成一个元组，然后各元组构成一个大zip
# 对于各元组中的两个图像
for la, lb in zip(lpA, lpB):
    # 取la或lb的尺寸皆可
    rows, dpt = la.shape
    # 利用np.hstack将这两个图像“一半一半”地拼接起来
    # 取la的左边一半和lb的右边一半拼成一个融合后的图，结果赋给ls
    ls = np.hstack((la[:, 0:dpt // 2], lb[:, dpt // 2:]))
    # 两个拉普拉斯金字塔各层图像融合后的结果赋给LS
    LS.append(ls)

# 用融合后的拉普拉斯金字塔重构出最终图像
# 初始化ls为融合后拉普拉斯金字塔的最高层
# 下面的循环结束后ls就是要求的最终结果图像
ls = LS[0]
for i in range(1, 8):
    # 每层图像先上采样，再和当前层的下一层图像相加，结果再赋给ls
    ls = cv2.pyrUp(ls)
    ls = cv2.add(ls, LS[i])

with_pyramid = lr + ls
# 不用金字塔融合，直接生硬地连接两幅原始图像
without_pyramid = np.hstack((A[:,:dpt//2],B[:,dpt//2:]))

# 对比一下结果
cv2.imshow("with_pyramid",with_pyramid)
cv2.imshow("without_pyramid",without_pyramid)
cv2.waitKey(0)
'''

#img_001=cv2.addWeighted(imge_1,0.5,imge_01,0.5,0)
#img_00 = cv2.addWeighted(sceneRadiance_CLHAE,0.5,sceneRadiance_UCM,0.3,0)
#print(imge_2.shape)

cv2.imshow("imge_1", imge_1)
cv2.imshow("imge_2", imge_2)
cv2.imshow("imge_01", imge_01)
cv2.imshow("imge_02", imge_02)
cv2.imshow("upscaled_UCM2", upscaled_UCM2)
cv2.imshow("upscaled_CLAHE2", upscaled_CLAHE2)

img_002 = cv2.addWeighted(imge_1, 0.5, imge_01, 0.5, 0)  #(336,474,3)
img_003 = cv2.addWeighted(imge_2, 0.5, imge_02, 0.5, 0)
img_004 = cv2.addWeighted(upscaled_CLAHE2, 0.5, upscaled_UCM2, 0.5, 0)

#上采样卷积对比度拉伸

upscaled_CLAHE_004 = cv2.pyrUp(img_004)
# blurred_CLAHE_004 = cv2.filter2D(upscaled_CLAHE_004, -1, gaussian_kernel)
"""
# 计算原始图像的最小和最大灰度值6
min_value = np.min(blurred_CLAHE_004)
max_value = np.max(blurred_CLAHE_004)
# 执行对比度拉伸
stretched_image = (blurred_CLAHE_004 - min_value) * (255.0 / (max_value - min_value))
# 将图像灰度值限制在0到255范围内
stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)
"""

stretched_image=RecoverCLAHE(upscaled_CLAHE_004)


#融合
img_005 = cv2.addWeighted(img_003, 1, stretched_image, 1, 0)
#上采样卷积对比度拉伸
#print(img_005.shape)
upscaled_UCM_005 = cv2.pyrUp(img_005)
# blurred_UCM_005 = cv2.filter2D(upscaled_UCM_005, -1, gaussian_kernel)

cv2.imshow("blurred_UCM_005", upscaled_UCM_005)
cv2.imshow("stretched_image", stretched_image)

#融合
#print(img_002.shape)
#print(upscaled_UCM_005.shape)
#new_size_3 = (335, 474, 3)
#resized_image_3 = np.resize(blurred_UCM_005, new_size_3)

# print("\n")
# print(img_002.shape)
# print(blurred_UCM_005.shape)
img_006 = cv2.addWeighted(img_002, 1, upscaled_UCM_005, 1, 0)

# img_006=img_002+upscaled_UCM_005
img_007 = cv2.addWeighted(sceneRadiance_CLHAE, 0.5, sceneRadiance_UCM, 0.5, 0)
cv2.imshow("img_007", img_007)
# height, width = img_006.shape[:2]
# new_size = (width // 2, height // 2)
# resized_img = cv2.resize(img_006, new_size, interpolation=cv2.INTER_LINEAR)

#print("111")
cv2.imshow("004", img_006)
cv2.waitKey(0)
cv2.destroyAllWindows()

