import cv2 as cv
import numpy as np


# 高斯金字塔和拉普拉斯金字塔
# 高斯金字塔：通过 reduce（高斯模糊+降采样）得到。公式：g2=reduce(g1)...
# 拉普拉斯金字塔：通过 expand（升采样+卷积）得到。公式：L1 = g1 - expand(g2))...

# pryDown(src, dst=None, dstsize=None, borderType=None):先对图像进行高斯平滑，再进行降采样
# 第三个参数指定降采样之后的目标图像的大小，默认为行列减少一半；若自己指定需满足以下条件
# |dstsize.width * 2 - src.cols| ≤ 2， |dstsize.height * 2 - src.rows| ≤ 2;即只能为行列减少一半后多一行或少一行

# pyrUp(src, dst=None, dstsize=None, borderType=None)：先对图像进行升采样，然后再进行高斯平滑
# 目标图像默认大小为行列扩大一倍

# L1 = g1 - expand(g2)) 的示例
def example(src):
    down = cv.pyrDown(src)
    cv.imshow("down", down)

    up = cv.pyrUp(down)
    cv.imshow("up", up)

    sub = cv.subtract(src, up)
    cv.imshow("sub", sub)


def pyramid_demo(image):
    level = 4   # 金字塔层数
    temp = image.copy()
    pyramid_images = [] # 存取图像的列表

    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)  
        cv.imshow("pyramid_down_"+str(i+1), dst)
        temp = dst.copy()
    return pyramid_images


def laplace_demo(image):  # 注：图片必须是满足2^n这种分辨率
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)

    for i in range(level-1, -1, -1):    # 层数由高到低递减
        if i-1 < 0:
            # 最底层与原图作差
            expand  = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("laplace_demo"+str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow("laplace_demo"+str(i), lpls)


src = cv.imread("img/lena.jpg")  # 读入图片放进src中
cv.imshow("src", src)  # 将src图片放入该创建的窗口中

example(src)
# pyramid_demo(src)
# laplace_demo(src)

cv.waitKey(0) # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
cv.destroyAllWindows()  # 关闭所有窗口