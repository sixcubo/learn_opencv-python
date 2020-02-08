import cv2 as cv
import numpy as np


# 百度百科：霍夫变换是一种特征检测(feature extraction)，被广泛应用在图像分析、计算机视觉以及数位影像处理。
# 霍夫变换是用来辨别找出物件中的特征，例如：线条。
# 他的算法流程大致如下，给定一个物件、要辨别的形状的种类，算法会在参数空间(parameter space)中执行投票
# 来决定物体的形状，而这是由累加空间(accumulator space)里的局部最大值(local maximum)来决定。

# 关于霍夫变换的相关知识可以看看这个博客：https://blog.csdn.net/kbccs/article/details/79641887

# 笛卡尔坐标的一个点对应霍夫空间的一条直线（反过来也成立），笛卡尔坐标上共线的点在霍夫空间对应的直线有一个交点。
# 将笛卡尔坐标上的数个点变换为霍夫空间的直线，直线会有多个交点，交点包含的直线越多（信号越强），这些直线对应到笛卡尔坐标的点就越可能共线（这就是“投票”）。
# 找到霍夫空间中信号最强的点，就可以找到对应笛卡尔坐标的直线。

def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3) # apertureSize是Sobel算子的大小（默认值为3）
    cv.imshow("canny", edges)

    # HoughLines(image, rho, theta, threshold, lines=None, srn=None, stn=None, min_theta=None, max_theta=None) -> lines
    # 返回值lines为霍夫空间的点集，对应笛卡尔坐标的直线集合
    # 每个点以（ρ,θ）的形式给出。ρ 的单位是像素，θ 的单位是弧度。
    # @image 是二值化图像，所以先进行二值化，或者进行 Canny 边缘检测
    # @rho 和 @theta 代表 ρ 和 θ 的精确度
    # @threshold 是阈值，只有累加其中的值高于阈值时才被认为是一条直线，也可以把它看成能检测到的直线的最短长度（以像素点为单位）。
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)

    print(lines.shape)  # n×1×2.检测出n条直线
    print(lines)

    for line in lines:
        # 获取点的坐标，根据此点可以找到原图的一条直线
        rho, theta = line[0]

        # 找出平面直角坐标系中的两个点（由极坐标转换得到）
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))

        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        # 通过两点画一条直线
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imshow("line_detection", image)


def line_detection_possible(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3) # apertureSize是Sobel算子的大小（默认值为3）

    minLineLength = 100 # line最小长度
    maxLineGap = 10     # line最大间隔
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)

    print(lines.shape)  # 
    print(lines)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow('hough_lines', image)


# Hough Circle 在xy坐标系中一点对应Hough坐标系中的一个圆，xy坐标系中圆上各个点对应Hough坐标系各个圆，
# 相加的一点，即对应xy坐标系中圆心
# 现实考量：Hough圆对噪声比较敏感，所以做hough圆之前要进行滤波，
# 基于效率考虑，OpenCV中实现的霍夫变换圆检测是基于图像梯度的实现，分为两步：
# 1. 检测边缘，发现可能的圆心候选圆心开始计算最佳半径大小
# 2. 基于第一步的基础上，从候选圆心开始计算最佳半径大小
def detection_circles(image):
    # 均值迁移，消除噪声，霍夫圆检测对噪声敏感
    dst = cv.pyrMeanShiftFiltering(image, 10, 100) 

    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

    # HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None) -> circles
    # @param image 8位，单通道，灰度图像
    # @param method 检测方法，目前唯一实现的方法只有 HOUGH_GRADIENT
    # @param dp 累加器分辨率和图像分辨率的反比
    #   这个参数允许创建一个比输入图像分辨率低的累加器。（这样做是因为有理由认为图像中存在的圆会自然降低到与图像宽高相同数量的范畴）。
    #   如果dp设置为1，表示霍夫空间与输入图像空间的大小一致；dp=2时霍夫空间是输入图像空间的一半，以此类推。dp的值不能比1小。
    # @param minDist 圆心之间的最小距离，如果检测到的两个圆心之间距离小于该值，则认为它们是同一个圆心
    # @param param1 用于Canny的边缘阀值上限，下限被置为上限的一半。这个值越大，检测圆边界时，要求的亮度梯度越大。
    # @param param2 在检测阶段中圆心的累加器阈值。这个值越小，越多错误的圆会被检测到
    # @param minRadius 所检测到的圆半径的最小值
    # @param maxRadius 所检测到的圆半径的最大值
    # 返回值circles包含多个圆向量，每个圆向量由三个值组成（圆心坐标，半径大小）
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=40, param2=30, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles)) # np.around 返回四舍五入后的值，可指定精度。

    print(circles.shape)    # 一行n列三通道（检测出n个圆）

    for i in circles[0, :]:  
        # 画圆
        # circle(img, center, radius, color, thickness=None, lineType=None, shift=None)
        cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2) 
        # 画出圆心（将半径指定为2）
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 3)
    cv.imshow('detected circles', image)


def main():
    # src = cv.imread("img/sudoku.png")
    src = cv.imread("img/chessboard.png")
    cv.imshow("demo",src)

    # line_detection(src)

    # line_detection_possible(src)

    img = cv.imread("img/circle.png")
    detection_circles(img)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()