import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# 灰度图像直方图
def plot_demo(image):
    # image.ravel() 将图像展开（将多维数组降位一维）。256为bins数量（组数），[0, 256]为统计范围（横坐标范围）
    plt.hist(image.ravel(), 256, [0, 256])  
    plt.show()


# rgb三通道直方图
def image_hist(image):
    color = enumerate(('blue', 'green', 'red')) # 枚举量
    for i, color in color:

        print(i, color)

        # 计算出直方图，calcHist(images, channels, mask, histSize(有多少个bins), ranges[, hist[, accumulate]]) -> hist
        # 得出的 hist 是一个 256x1 的一维数组，每一个值代表了与该灰度值对应的像素点数目。
        hist = cv.calcHist(image, [i], None, [256], [0, 256])
        print(hist.shape)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])  # 设定 x轴范围
    plt.show()


# 直方图均衡化，用于增强图像对比度
def equalHist_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    # 全局直方图均衡化
    dst = cv.equalizeHist(gray)
    cv.imshow("equalHist_demo", dst)

    # CLAHE 局部直方图均衡化
    # clipLimit 对比阈值限制, tileGridSize 划分方格数
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_dst = clahe.apply(gray)
    cv.imshow("clahe", clahe_dst)


# 利用直方图比较相似性
# 创建直方图
def create_rgb_demo(image):

    h, w, c = image.shape

    # rgbHist是一个一维数组，每一个值代表了与该种颜色对应的像素点数目。
    # 降为一维，每个通道有16个bins，三个通道一共组合出16*16*16个
    rgbHist = np.zeros([16*16*16, 1], np.float32)

    # 每个通道颜色取值为0到255，每个通道有16个bins，所以每个bins大小为256/16
    bsize = 256 / 16    

    # 遍历每个像素
    for row in range(h):
        for col in range(w):
            # 获取bgr值
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]

            # 三个维度b乘16*16，g乘16，r乘1相当于图片降为一维,确定此像素颜色一维表示的index
            index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
            rgbHist[np.int(index), 0] += 1
            # rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1

    return rgbHist


# 几种不同的比较方法，用巴氏和相关性比较好
def hist_compare(image1, image2):
    hist1 = create_rgb_demo(image1)
    hist2 = create_rgb_demo(image2)

    match1 = cv.compareHist(hist1, hist2, method=cv.HISTCMP_BHATTACHARYYA)  # 巴氏距离，越小越相似
    match2 = cv.compareHist(hist1, hist2, method=cv.HISTCMP_CORREL)         # 相关性，越接近1越相似
    match3 = cv.compareHist(hist1, hist2, method=cv.HISTCMP_CHISQR)         # 卡方，越小越相似

    print("巴式距离：%s, 相关性：%s, 卡方：%s"%(match1, match2, match3))


# 2d直方图
def hist2d_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # 转换为hsv
    # 计算h和s通道的2d直方图，h通道范围0到179，s通道范围0到255
    cv.imshow("", hsv)
    hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    #cv.imshow("hist2d", hist)
    plt.imshow(hist)
    plt.title("hist2d")
    plt.show()


# 反向投影
def back_prijection_demo():
    sample = cv.imread("img/sample.jpg")
    target = cv.imread("img/target.jpg")

    cv.imshow("sample", sample)
    cv.imshow("target", target)
    
    sample_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    # 可通过调整参数histSize，改善投影碎片化现象，因为减小histSize，bins的大小变大
    # sample_hist = cv.calcHist([sample_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    sample_hist = cv.calcHist([sample_hsv], [0, 1], None, [32, 48], [0, 180, 0, 256])

    # 归一化函数normalize
    # src：输入数组，
    # dst：输出数组，支持原地运算，
    # alpha：range normalization模式的最小值，
    # beta：range normalization模式的最大值，不用于norm normalization(范数归一化)模式，
    # normType：归一化的类型，一般较常用 NORM_MINMAX (数组的数值被平移或缩放到一个指定的范围，线性归一化)。
    cv.normalize(sample_hist, sample_hist, 0, 255, cv.NORM_MINMAX)

    dst = cv.calcBackProject([target_hsv], [0, 1], sample_hist, [0, 180, 0, 256], 1)

    cv.imshow("dst", dst) 


src = cv.imread("img/home.jpg")  # 读入图片放进src中
cv.namedWindow("demo")  # 创建窗口
cv.imshow("demo", src)  # 将src图片放入该创建的窗口中

# plot_demo(src)
# image_hist(src)

# equalHist_demo(src)
# image1 = cv.imread("img/rice.png")
# image2 = cv.imread("img/noise_rice.png")

# create_rgb_demo(image1)
# cv.imshow("image1", image1)
# cv.imshow("image2", image2)
# hist_compare(image1=image1, image2=image2)

# hist2d_demo(src)
back_prijection_demo()

cv.waitKey(0) # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
cv.destroyAllWindows()  # 关闭所有窗口