import cv2 as cv
import numpy as np


"""
在图像处理技术中，有一些的操作会使图像的形态发生改变，这些操作一般称之为形态学操作（phology）。
数学形态学是基于集合论的图像处理方法，最早出现在生物学的形态与结构中，
图像处理中的形态学操作用于图像预处理操作（去噪，形状简化）、图像增强（骨架提取，细化，凸包及物体标记）、物体背景分割及物体形态量化等场景中，
形态学操作的对象是二值化图像。
形态学操作中包括腐蚀，膨胀，开操作，闭操作等。其中腐蚀，膨胀是许多形态学操作的基础。
"""

# 膨胀是图像中的高亮部分进行膨胀，领域扩张，效果图拥有比原图更大的高亮区域。可以平滑对象，填充或填充对象间的距离
# 腐蚀是图像中的高亮部分被腐蚀掉，领域缩减，效果图拥有比原图更小的高亮区域。平滑对象边缘，弱化或分割对象间的半岛型连接,去除噪点


# 腐蚀原理：局部最小值(与膨胀相反)；
# ①定义一个卷积核B，核可以是任何的形状和大小，且拥有一个单独定义出来的参考点-锚点(anchorpoint)；通常和为带参考点的正方形或者圆盘，可将核称为模板或掩膜；
# ②将核B与图像A进行卷积，计算核B覆盖区域的像素点最小值；
# ③将这个最小值赋值给参考点指定的像素；因此，图像中的高亮区域逐渐减小。
def erode_bin(image):
    # 二值化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    # 关于卷积核B，一般配合getStructuringElement()使用；
    # getStructuringElement(shape, ksize, anchor=None)：返回指定形状和尺寸的结构元素
    # @shape：元素的形状，可以是 #MorphShapes中的之一。矩形 MORPH_RECT、十字形 MORPH_CROSS、椭圆形 MORPH_ELLIPSE;
    # @ksize: 结构元素的尺寸
    # @anchor：元素内的锚点。有默认值(-1, -1)，指定anchor在element的中心。
    #   只有十字形元素依靠anchor，其他情况anchor只影响形态学运算结果的偏移。
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.erode(binary, kernel=kernel)
    cv.imshow("erode_demo", dst)


# 膨胀原理：求局部最大值;
# ①定义一个卷积核B，核可以是任何的形状和大小，且拥有一个单独定义出来的参考点-锚点(anchorpoint)；通常和为带参考点的正方形或者圆盘，可将核称为模板或掩膜；
# ②将核B与图像A进行卷积，计算核B覆盖区域的像素点最大值；
# ③将这个最大值赋值给参考点指定的像素；因此，图像中的高亮区域逐渐增长。
def dilate_bin(image):
    # 二值化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.dilate(binary, kernel=kernel)
    cv.imshow("dilate_demo", dst)


def erode_dilate_bgr(img):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    dst_dilate = cv.dilate(img, kernel=kernel)
    dst_erode = cv.erode(img, kernel=kernel)
    cv.imshow("dilate", dst_dilate)
    cv.imshow("erode", dst_erode)


def main():
    # src = cv.imread("img/01.jpg")
    # cv.imshow("src", src)

    # erode_bin(src)
    # dilate_bin(src)

    # 彩色图像腐蚀，膨胀
    img = cv.imread("img/lena.jpg")
    cv.imshow("img", img)
    erode_dilate_bgr(img)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()