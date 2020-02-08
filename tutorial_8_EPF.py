import cv2 as cv
import numpy as np

# 双边滤波
# 同时考虑空间与信息和灰度相似性，达到保边去噪的目的
# 双边滤波的核函数是空间域核与像素范围域核的综合结果：
# 在图像的平坦区域，像素值变化很小，对应的像素范围域权重接近于1，此时空间域权重起主要作用，相当于进行高斯模糊；
# 在图像的边缘区域，像素值变化很大，像素范围域权重变大，从而保持了边缘的信息。
def bi_demo(image):  # bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None)
    dst = cv.bilateralFilter(image, 0, 100, 15)  # 高斯双边
    cv.imshow("bi_demo", dst)


# 均值迁移
# 原理：对于给定的一定数量样本，任选其中一个样本，以该样本为中心点划定一个圆形区域，
# 求取该圆形区域内样本的质心，即密度最大处的点，再以该点为中心继续执行上述迭代过程，直至最终收敛。

# 可以利用均值偏移实现彩色图像分割，但这个函数严格来说并不是图像的分割，而是图像在色彩层面的平滑滤波，
# 它可以中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域。要达到图像分割需使用漫水填充。
# 详见：https://blog.csdn.net/dcrmg/article/details/52705087
def shift_demo(image):  
    # pyrMeanShiftFiltering(src, sp, sr, dst=None, maxLevel=None, termcrit=None)
    # @param src 8位，3通道图像
    # @param dst 与原图有相同的格式
    # @param sp 漂移物理空间半径大小
    # @param sr 漂移色彩空间半径大小
    # @param maxLevel Maximum level of the pyramid for the segmentation.
    # @param termcrit Termination criteria: when to stop meanshift iterations.
    # 关键参数是sp和sr的设置，二者设置的值越大，对图像色彩的平滑效果越明显，同时函数耗时也越多
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow("shift_demo", dst)


src = cv.imread("img/CrystalLiu1.jpg")  # 读入图片放进src中
cv.namedWindow("Crystal Liu")  # 创建窗口
cv.imshow("Crystal Liu", src)  # 将src图片放入该创建的窗口中

bi_demo(src) 

shift_demo(src)

cv.waitKey(0) # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
cv.destroyAllWindows()  # 关闭所有窗口