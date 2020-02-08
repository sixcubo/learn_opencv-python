import cv2 as cv
import numpy as np


def object_measure(image):
    # 二值化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    print("threshold value: %s"%ret, "\n")
    cv.imshow("binary image", binary)

    # 轮廓检测
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        # 绘制轮廓（黄色）
        cv.drawContours(image, contours, i, (0, 255, 255), 1)  

        # 计算轮廓面积
        # contourArea(contour, oriented=None)
        area = cv.contourArea(contour)  
        print("contour area [%d]:"%i, area)

        # 计算轮廓周长，第二参数closed（boolean型）用来指定对象的形状是否为闭合的。
        perimeter = cv.arcLength(contour, True)
        print("contour perimeter [%d]:"%i, perimeter)

        # 获取轮廓外接矩形，返回值为矩形的坐标和宽高
        x, y, w, h = cv.boundingRect(contour)  
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # 计算矩阵宽高比
        rate = min(w, h)/max(w, h)  
        print("rectangle rate [%d]"%i,rate)
        
        # 计算图像矩，得到的矩以一个dict的形式返回
        mm = cv.moments(contour)  

        # 根据公式，计算出对象的重心
        cx = mm['m10']/mm['m00']
        cy = mm['m01']/mm['m00']

        # 用实心圆画出重心
        cv.circle(image, (np.int(cx), np.int(cy)), 2, (0, 255, 255), -1)  

        print("\n")

    cv.imshow("measure object", image)


def contour_approx(image):
    # 二值化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value: %s" % ret)
    cv.imshow("binary image", binary)

    # 因为后面需要在二值图像上绘制轮廓和点（轮廓和点为三通道），先转化为BGR图像
    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

    # 轮廓检测
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):

        # 把一个连续光滑曲线折线化，对图像轮廓点进行多边形拟合
        # approxPolyDP(curve, epsilon, closed, approxCurve=None) -> approxCurve
        # @param curve 输入
        # @param approxCurve 输出
        # @param epsilon 逼近精度. 原始曲线与近似曲线之间的最大距离.
        # @param closed 如果为True，则近似曲线是闭合的（它的第一个和最后的顶点连接）。
        epsilon = 0.01 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        
        print(approx.shape) # m×1×2. 每个轮廓拟合出m个点，每个点用横纵坐标值确定

        # 绘制拟合出的点
        # 参数contourIdx为-1，绘制该轮廓所有的点
        cv.drawContours(dst, approx, -1, (255, 0, 0), 10)

        # 如果点数为4，为矩形，用黄色点画出
        if approx.shape[0] == 4:
            cv.drawContours(dst, contours, i, (0, 255, 255), 2)  

        # 如果点数为3，为三角形，用红色点画出
        if approx.shape[0] == 3:
            cv.drawContours(dst, contours, i, (0, 0, 255), 2)  

        # 如果点数大于10，为圆形，
        if approx.shape[0] > 10:
             cv.drawContours(dst, contours, i, (255, 255, 0), 2)  


    cv.imshow("dst", dst)


def main():
    # src = cv.imread("img/handwriting.jpg")
    # cv.imshow("src",src)
    # measure_object(src)

    img = cv.imread("img/approx.png")
    contour_approx(img)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()