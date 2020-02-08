import cv2 as cv
import numpy as np


# 分水岭算法
# Any grayscale image can be viewed as a topographic surface where high intensity denotes peaks and hills while low intensity denotes valleys.
# You start filling every isolated valleys (local minima) with different colored water (labels). 
# As the water rises, depending on the peaks (gradients) nearby, water from different valleys, 
# obviously with different colors will start to merge. To avoid that, you build barriers in the locations where water merges.
# You continue the work of filling water and building barriers until all the peaks are under water. 
# Then the barriers you created gives you the segmentation result. This is the "philosophy" behind the watershed. 
#
# 任何一副灰度图像都可以被看成拓扑平面，灰度值高的区域可以被看成是山峰，灰度值低的区域可以被看成是山谷。
# 用不同颜色的水（标签）填充每个孤立的山谷（局部最小值）。随着水位的升高，不同山谷的水（颜色不同）会相遇汇合，
# 为了避免汇合，需要在水汇合的地方构建起障碍。继续不停的填充水，并不停的构建障碍，直到所有的山峰都被水淹没。
# 那么构建好的障碍就是对图像分割的结果。这就是分水岭算法背后的哲理。


# But this approach gives you oversegmented result due to noise or any other irregularities in the image. 
# So OpenCV implemented a marker-based watershed algorithm where you specify which are all valley points are to be merged and which are not. 
# It is an interactive image segmentation. What we do is to give different labels for our object we know. 
# Label the region which we are sure of being the foreground or object with one color (or intensity), 
# label the region which we are sure of being background or non-object with another color 
# and finally the region which we are not sure of anything, label it with 0. 
# That is our marker. Then apply watershed algorithm. 
# Then our marker will be updated with the labels we gave, and the boundaries of objects will have a value of -1.
#
# 但是由于噪声或者图像中其他不规律的因素，这种方法通常会得到过度分割的结果。
# 所以 OpenCV 采用了基于标记（marker-based）的分水岭算法，在这种算法中你要指定哪些山谷汇合，哪些不会。
# 这是一种交互式的图像分割。我们要做的就是给我们已知的对象打上不同的标签。
# 用一个颜色标记那些我们确定是前景或物体的区域。用另一个颜色标记那些我们确定是背景或不是物体的区域。
# 最后，用 0 标记那些我们不确定的区域。然后应用分水岭算法。
# 每一次填充水，我们的标签会被自动更新，最后得到的物体边界的值为 -1。
#
# 来自官方文档：https://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html


# 基于距离的分水岭分割流程：
# 输入图像->灰度->二值->距离变换->寻找种子->生成marker->分水岭变换->输出图像


# 距离变换
# 计算一个图像中非零像素点到最近的零像素点的距离，也就是到零像素点的最短距离。
# 一个最常见的距离变换算法就是通过连续的腐蚀操作来实现，腐蚀操作的停止条件是所有前景像素都被完全腐蚀。
# 这样根据腐蚀的先后顺序，我们就得到各个前景像素点到前景中心像素点的距离。
# 根据各个像素点的距离值，设置为不同的灰度值。这样就完成了二值图像的距离变换


def watershed_demo(image):

    # 均值迁移
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    # 灰度
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    # 确定背景
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel=kernel, iterations=2) 
    sure_bg = cv.dilate(opening, kernel, iterations=3)  # 膨胀
    cv.imshow("sure_bg", sure_bg)
    
    # 确定前景
    # distanceTransform(src, distanceType, maskSize, dst=None, dstType=None)：距离变换
    # @param distanceType 选取距离的类型。CV_DIST_L1, CV_DIST_L2 , CV_DIST_C等
    # @param maskSize 距离变换掩膜的大小
    # @param dst 输出保存了每一个点与最近的零点的距离信息，图像上越亮的点，代表了离零点的距离越远。
    #   根据这个性质，经过简单的运算，可用于细化字符的轮廓和查找物体质心（中心）。
    dist = cv.distanceTransform(opening, cv.DIST_L2, 3)
    cv.normalize(dist, dist, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX)   # 归一化
    # cv.imshow("", dist)
    ret, sure_fg = cv.threshold(dist, 0.6*dist.max(), 255, cv.THRESH_BINARY)
    cv.imshow("sure_fg", sure_fg)

    # 确定不确定区域
    sure_fg = np.uint8(sure_fg) # 由 float32 转化为 uint8
    unsure = cv.subtract(sure_bg, sure_fg)
    cv.imshow("unsure", unsure)

    #连通分量
    ret, markers1 =cv.connectedComponents(sure_fg)  # markers1.dtype == int32
    print(ret)

    # watershed transform
    markers2 = markers1 + 1
    markers2[unsure==255] = 0   # 标记不确定区域
    markers3 = cv.watershed(image, markers=markers2)
    image[markers3 == -1] =[0, 0, 255]  # 画出物体边界

    cv.imshow("result", image)

def main():
    src = cv.imread("img/circle.png")
    cv.imshow("src",src)

    watershed_demo(src)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()