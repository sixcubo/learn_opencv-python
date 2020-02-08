# 光流(optical flow)法是利用图像序列中像素在时间域上的变化以及相邻帧之间的相关性，
# 来找到上一帧跟当前帧之间存在的对应关系，从而计算出相邻帧之间物体的运动信息的一种方法。
# 所谓光流就是瞬时速率，在时间间隔很小（比如视频的连续前后两帧之间）时，也等同于目标点的位移。

# 在计算机视觉的空间中，计算机所接收到的信号往往是二维图片信息。由于缺少了一个维度的信息，所以其不再适用以运动场描述。
# 光流场（optical flow）就是用于描述三维空间中的运动物体表现到二维图像中，所反映出的像素点的运动向量场。
# 当描述部分像素时，称为：稀疏光流
# 当描述全部像素时，称为：稠密光流

# 光流法的前提假设：
# 相邻帧之间亮度恒定；
# 相邻帧之间取时间连续或者运动变化微小；（如果有大的运动时，可以使用图像金字塔解决）
# 同一子图像中像素点具有相同的运动。

import cv2 as cv
import numpy as np

# L-K光流法
# Lucas–Kanade光流算法是一种两帧差分的光流估计算法
# 最初于1981年提出，该算法假设在一个小的空间邻域内运动矢量保持恒定，使用加权最小二乘法估计光流。
# 由于该算法应用于输入图像的一组点上时比较方便，因此被广泛应用于稀疏光流场。


def LK_demo():  # 使用函数cv2.goodFeatureToTrack()来确定要跟踪的点，然后使用Lucas-Kanade算法迭代跟踪这些角点。

    # 读取第一帧图像，转为灰度
    cap = cv.VideoCapture('img/slow.mp4')
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    # 创建随机颜色
    color = np.random.randint(0, 255, (100, 3))

    # 创建字典，存放goodFeaturesToTrack()的部分参数
    feature_params = dict(maxCorners=100, qualityLevel=0.3,
                          minDistance=7, blockSize=7)

    # 确定图像上的强角点
    # 函数原型：goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance,
    #               corners=None, mask=None, blockSize=None, useHarrisDetector=None, k=None)
    # @param image 输入 8位或浮点型32位单通道图像
    # @param corners 角点检测结果的输出向量
    # @param maxCorners 最大角点数目，如果检测出更多角点，优先返回较强的。如果 maxCorners <= 0，代表最大角点数目无限制，返回所有检测结果
    # @param qualityLevel 参数指出图像角点的最小可接受质量（小于1.0的正数，一般在0.01-0.1之间）
    # @param minDistance 被返回的角点之间的最小可能欧氏距离
    # @param mask 指定ROI
    # @param blockSize 计算协方差矩阵时的窗口大小
    # @param useHarrisDetector 参数指定是否使用Harris检测器
    # @param k Free parameter of the Harris detector.
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # 创建与old_frame有相同shape的全0矩阵
    all_zero = np.zeros_like(old_frame)

    # 创建字典，存放calcOpticalFlowPyrLK()的部分参数
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
        cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:

        # 读取新的帧，转为灰度
        ret, new_frame = cap.read()
        new_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

        # 基于角点特征的金字塔LK光流跟踪算法
        # 函数原型：calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status=None, err=None,
        #               winSize=None, maxLevel=None, criteria=None, flags=None, minEigThreshold=None) -> nextPts, status, err
        # @param prevImg 前一帧8位输入图像或 由 buildOpticalFlowPyramid 构造的金字塔
        # @param nextImg 后一帧8位输入图像或 和 prevImg 有相同 size 和 type 的金字塔
        # @param prevPts 计算光流所需要的输入2D点矢量，点坐标必须是单精度浮点数
        # @param nextPts 输出2D点矢量(也是单精度浮点数坐标)，点矢量中包含的是在后一帧图像上计算得到的输入特征新位置
        # @param status 输出状态矢量(元素是无符号char类型，uchar)。如果角点发现光流，则矢量元素置为1，否则，为0
        # @param err 输出误差矢量
        # @param winSize 每个金字塔层的搜索窗大小
        # @param maxLevel 金字塔的最大层数；如果置0，不使用金字塔(即单层)；如果置1，金字塔2层，以此类推；
        #                 如果金字塔被传递给输入，算法将使用与输入相同的层数，但不超过maxLevel
        # @param criteria 指定搜索算法收敛迭代的类型
        # @param flags operation flags
        # @param minEigThreshold 算法计算的光流等式的2x2常规矩阵的最小特征值
        p1, st, err = cv.calcOpticalFlowPyrLK(
            old_gray, new_gray, p0, None,  **lk_params)

        print(st)
        print(st == 1)

        # 提取存在光流的角点，（shape==n×2，即每行为一个角点坐标）
        old_good = p0[st == 1]
        new_good = p1[st == 1]

        # zip()函数将可迭代对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        corners_zip = zip(new_good, old_good)

        for i, (new, old) in enumerate(corners_zip):
            # 获取坐标值
            old_x, old_y = old
            new_x, new_y = new

            # 在画面上，画出角点
            new_frame = cv.circle(new_frame, (new_x, new_y),
                                  5, color[i].tolist(), -1)

            # 在all_zero上，利用前后两角点画出移动轨迹
            all_zero = cv.line(all_zero, (new_x, new_y), (old_x, old_y),
                               color[i].tolist(), 2)

        # 将移动轨迹添加到画面上
        img = cv.add(new_frame, all_zero)

        cv.imshow("frame", img)

        k = cv.waitKey(30) & 0xFF
        if k == 27:
            break

        # 以新代旧
        old_gray = new_gray.copy()
        p0 = new_good.reshape(-1, 1, 2)  # 如果置为-1，Numpy会根据剩下的维度自动计算出shape属性值

    cv.destroyAllWindows()
    cap.release()


"""
Lucas-Kanade 法是计算一些特征点的光流。OpenCV 还提供了一种计算稠密光流的方法。
它会图像中的所有点的光流。这是基于 Gunner_Farneback 的算法 （2003 年）。
结果是一个带有光流向量 （u，v）的双通道数组。通过计算我们能得到光流的大小和方向。
使用颜色对结果进行编码以便于更好的观察。
方向对应于 H（Hue）通道，大小对应 于 V（Value）通道
:return:
"""


def Farneback_demo():
    # 读取第一帧
    cap = cv.VideoCapture("img/vtest.avi")
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # 将S通道置为255

    while True:

        # 读取新的帧
        ret, frame2 = cap.read()
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # 用Gunnar_Farneback算法计算稠密光流
        # 为了解决孔径问题，函数中引入了图像金字塔
        # 形象理解：从小孔中观察一块移动的黑色幕布观察不到任何变化。但实际情况是幕布一直在移动中
        # 解决方案：从不同尺度（图像金字塔）上对图像进行观察，由高到低逐层利用上一层已求得信息来计算下一层信息
        # 函数原型：calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) -> flow
        # @param prev 前一帧输入图像，8位单通道
        # @param next 后一帧输入图像，和prev有相同的size
        # @param flow 输出的光流，和prev有相同的size并且type为CV_32FC2.
        # @param pyr_scale 金字塔上下两层之间的尺度关系
        # @param levels 金字塔层数（包括初始图像）
        # @param winsize 均值窗口大小，越大越能denoise并且能够检测快速移动目标，但会引起模糊运动区域
        # @param iterations 算法在每层金字塔的迭代次数
        # @param poly_n 像素领域大小，用于查找每个像素的多项式展开，一般为5，7等
        # @param poly_sigma 高斯标准差，用来作为平滑多项式展开的基础导数；当poly_n=5，可置poly_sigma=1.1，poly_n=7, 可置poly_sigma=1.5. .
        # @param flags operation flags，主要包括OPTFLOW_USE_INITIAL_FLOW和OPTFLOW_FARNEBACK_GAUSSIAN
        flow = cv.calcOpticalFlowFarneback(
            prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 笛卡尔坐标转换到极坐标
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        hsv[..., 0] = ang / (2 * np.pi) * 180  # 设置H通道（范围0~180）
        hsv[..., 2] = cv.normalize(
            mag, None, 0, 255, cv.NORM_MINMAX)   # 设置V通道（0~255）

        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        dst = cv.add(rgb, frame2)
        cv.imshow('frame2', dst)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        # 以新代旧
        prvs = next

    cap.release()
    cv.destroyAllWindows()


def main():

    # LK_demo()

    Farneback_demo()

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
