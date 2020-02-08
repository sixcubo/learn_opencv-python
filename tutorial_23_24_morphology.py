import cv2 as cv
import numpy as np


"""
开运算:先腐蚀再膨胀。用来去除噪声。
闭运算:先膨胀再腐蚀。用来填充前景物体中的小洞，或者前景物体上的小黑点。

开闭操作作用：
1. 去除小的干扰块——开操作 
2. 填充闭合区域——闭操作 
3. 水平或垂直线提取,调整kernel的row，col值差异。比如：采用开操作，kernel为(1, 15),提取垂直线，kernel为(15, 1),提取水平线，
"""

"""
其他形态学操作：
顶帽：原图像与开操作之间的差值图像
黑帽：闭操作与原图像直接的差值图像
形态学梯度：一幅图像膨胀与腐蚀的差值。 结果看上去就像前景物体的轮廓
    基本梯度：膨胀后图像减去腐蚀后图像得到的差值图像。
    内部梯度：用原图减去腐蚀图像得到的差值图像。
    外部梯度：膨胀后图像减去原图像得到的差值图像。
"""


def open_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
   
    # # 去除噪点
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # 将ksize设置为直线型可以提取直线。
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 1))
    print(kernel.shape) # ksize=(15,1)，宽15高1，所以shape=(1,15)

    # 形态学变换函数 morphologyEx()
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel=kernel)
    cv.imshow("open_demo", dst)


def close_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    # 填充闭合区域
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 25))
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel=kernel)
    cv.imshow("open_demo", dst)

# 顶帽，黑帽
def tophat_balckhat(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)

    cv.imshow("top_hat", tophat)
    cv.imshow("black_hat", blackhat)


def gradient(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    # 基本梯度
    dst = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel=kernel)

    # 内部梯度，外部梯度
    erodeImg = cv.erode(binary, kernel)
    dilateImg = cv.dilate(binary, kernel)
    dst1 = cv.subtract(binary, erodeImg)   # 内部梯度：用原图减去腐蚀图像得到的差值图像。
    dst2 = cv.subtract(dilateImg, binary)  # 外部梯度：膨胀后图像减去原图像得到的差值图像。

    cv.imshow("basic gradient", dst)
    cv.imshow("internal gradient", dst1)
    cv.imshow("external gradient", dst2)



def main():
    src = cv.imread("img/open_close.png")
    src2 = cv.imread("img/lena.jpg")

    # open_demo(src)
    # close_demo(src)

    # tophat_balckhat(src2)

    gradient(src2)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()