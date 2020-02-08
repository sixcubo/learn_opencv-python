import cv2 as cv
import numpy as np


# 轮廓发现，基于边缘提取寻找对象轮廓

# 轮廓可以简单认为成将连续的点（连着边界）连在一起的曲线，具有相同的颜色或者灰度。
# 轮廓在形状分析和物体的检测和识别中很有用。
# 为了更加准确，要使用二值化图像。在寻找轮廓之前，要进行阈值化处理或者 Canny 边界检测。
# 查找轮廓的函数会修改原始图像。如果在找到轮廓之后还想使用原始图像的话，应该先将原始图像存储到其他变量中。
# 在 OpenCV 中，查找轮廓就像在黑色背景中找白色物体。要找的物体是白色，而背景是黑色。


# 二值化
def binary(image):
    dst = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, bin = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary image", bin)

    return bin


# canny边缘
def cannyEdge(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

    # edge = cv.Canny(grad_x, grad_y, 30, 150)
    edge = cv.Canny(gray, 50, 150)

    cv.imshow("edge", edge)

    return edge

 
def contours(image):
    
    # 图像二值化
    dst = binary(image)

    # 边缘提取
    # dst = cannyEdge(image)

    # findContours(image, mode, method, contours=None, hierarchy=None, offset=None) -> contours, hierarchy 
    # @image 8位单通道图像，可使用图像二值化或边缘提取的结果作为原图像
    # @mode 轮廓检索模式
    #   • CV_RETR_EXTERNAL - 只检索最外面的轮廓  
    #   • CV_RETR_LIST - 检索所有的轮廓，不建立任何层次关系（存在list中）
    #   • CV_RETR_CCOMP - 提取所有轮廓，并且将其组织为两层: 顶层是各部分的外部边界，第二层是空洞的边界
    #   • CV_RETR_TREE - 检索所有的轮廓，并重构嵌套轮廓的整个层次
    # @method 轮廓近似方法
    #   边缘近似方法（除了CV_RETR_RUNS使用内置的近似，其他模式均使用此设定的近似算法）。可取值如下：
    #   CV_CHAIN_CODE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
    #   CV_CHAIN_APPROX_NONE：将所有的连码点，转换成点。
    #   CV_CHAIN_APPROX_SIMPLE：压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。
    #   CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS：使用the flavors of Teh-Chin chain近似算法的一种。
    #   CV_LINK_RUNS：通过连接水平段的1，使用完全不同的边缘提取算法。使用CV_RETR_LIST检索模式能使用此方法。
    # @contours 检测到的轮廓
    # @hierarchy 包含关于图像的拓扑信息
    contours, hierarchy= cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        # drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
        # @contours 轮廓信息
        # @contourIdx 轮廓的索引（在绘制独立轮廓是很有用）。如果值是复数，将绘制所有轮廓。
        # @thickness 线宽，值为-1时对轮廓内部进行填充
        cv.drawContours(image, contours, i, (0, 0, 255), 2)  # 2为像素大小，-1时填充轮廓
        print(i)
    cv.imshow("detect contours", image)


def main():
    src = cv.imread("img/circle.png")
    cv.imshow("demo",src)

    contours(src)
    
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()