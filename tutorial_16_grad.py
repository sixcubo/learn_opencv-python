import cv2 as cv
import numpy as np


# 图像梯度（由x,y方向上的偏导数和偏移构成），一阶导数（sobel算子），二阶导数（Laplace算子）
# 一阶导数的极大值，二阶导数的零点，可用于求解图像边缘
# 一阶偏导在图像中为一阶差分，再变成算子（即权值）与图像像素值乘积相加，二阶同理


# Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None)
# src，输入图像
# ddepth，输出图像的深度，支持如下src.depth()和ddepth的组合：
#   若src.depth() = CV_8U, 取ddepth =-1/CV_16S/CV_32F/CV_64F
#   若src.depth() = CV_16U/CV_16S, 取ddepth =-1/CV_32F/CV_64F
#   若src.depth() = CV_32F, 取ddepth =-1/CV_32F/CV_64F
#   若src.depth() = CV_64F, 取ddepth = -1/CV_64F
#   其中 ddepth为-1时， 输出图像将和输入图像有相同的深度。输入8位图像则会截取顶端的导数。
# dx，x方向上的差分阶数。
# dy，y方向上的差分阶数。
# dst，目标图像，有和源图片一样的尺寸和类型。
# ksize，Sobel核的大小，只能为1，3，5或7。
# scale，计算导数值时可选的缩放因子，默认值是1，即不应用缩放。
# delta，结果在存入目标图之前可选的delta值，默认为0
# borderType，边界模式（边界像素外推方法），默认值为BORDER_DEFAULT。这个参数可以在官方文档中borderInterpolate处得到更详细的信息。
def sobel_demo(image):
    
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)   # 防止导数值被截断，使用CV_32F
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)

    # 当内核大小为 3 时, 以上Sobel内核可能产生比较明显的误差(毕竟，Sobel算子只是求取了导数的近似值)。
    # 为解决这一问题，OpenCV提供了 Scharr 函数，但该函数仅作用于大小为3的内核。该函数的运算与Sobel函数一样快，但结果却更加精确。
    # grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    # grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)

    # convertScaleAbs(src, dst=None, alpha=None, beta=None):计算绝对值，并将结果转换为 unsigned 8-bit type
    # src 输入数组, dst 输出数组, alpha 乘数因子, beta 偏移量
    gradx = cv.convertScaleAbs(grad_x)  
    grady = cv.convertScaleAbs(grad_y)

    # addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None)：
    # 计算两个图像的加权和， dst = src1*alpha + src2*beta + gamma
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)

    # 通过图像可以看出， gradx在x方向的边缘更明显，grady在y方向的边缘更明显
    cv.imshow("gradx", gradx)
    cv.imshow("grady", grady)
    cv.imshow("gradient", gradxy)


# Laplace算子
def laplace_demo(image):  
    dst = cv.Laplacian(image,cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("laplace_demo", lpls)


# 自定义卷积核
def custom_laplace(image):
    # 原Laplace算子的卷积核为 np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("custom_laplace", lpls)


def main():
    src = cv.imread("img/lena.jpg")
    cv.imshow("lena",src)

    sobel_demo(src)
    #laplace_demo(src)
    # custom_laplace(src)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()