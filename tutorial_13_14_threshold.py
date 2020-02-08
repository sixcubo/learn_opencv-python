import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 二值图像就是将灰度图转化成黑白图，没有灰，在一个值之前为黑，之后为白
# 有全局和局部两种
# 在使用全局阈值时，我们就是给出一个值来做阈值，那我们怎么知道我们选取的这个数的好坏呢？答案就是不停的尝试。
# 如果是一副双峰图像（简单来说双峰图像是指图像直方图中存在两个峰）呢？
# 我们岂不是应该在两个峰之间的峰谷选一个值作为阈值？这就是 Otsu 二值化要做的。
# 简单来说就是对一副双峰图像自动根据其直方图计算出一个阈值。
# （对于非双峰图像，这种方法 得到的结果可能会不理想）。


# 全局阈值化
def threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 第一个参数是原图像，原图像应该是灰度图。
    # 第二个参数是用来对像素值进行分类的阈值。
    # 第三个参数是分配给超过阈值的像素值的最大值
    # 第四个参数是阈值化的方法，见threshold_simple()。
    # 若不指定第二个参数，可以用阈值化方法按位或THRESH_OTSU或THRESH_TRIANGLE来自动计算阈值
    # 使用按位或运算符的原因：THRESH_OTSU的值为1000，THRESH_TRIANGLE的值为10000
    # 该方法返回两个输出。第一个是使用的阈值，第二个输出是经过阈值处理的图像。
    retval, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值化处理，并且阈值自动计算
    print("threshold value: %s"%retval)
    cv.imshow("threshold_demo", binary)


# 不同阈值化的方法
def threshold_simple(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')  # 将图像按2x3铺开
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


# 前面我们使用全局阈值，整幅图像采用同一个数作为阈值。
# 这并不适应与所有情况，尤其是当同一幅图像上的不同部分的具有不同亮度时。
# 这种情况下我们需要采用自适应阈值。此时的阈值是根据图像上的 每一个小区域计算与其对应的阈值。
# 因此在同一幅图像上的不同区域采用的是不同的阈值，从而使我们能在亮度不同的情况下得到更好的结果。

# 除上述参数外，函数cv.adaptiveThreshold还有三个输入参数，返回值只有一个
# adaptiveMethod决定阈值如何计算:
#     ADAPTIVE_THRESH_MEAN_C: 阈值是邻域面积减去常数C的平均值。
#     cv.ADAPTIVE_THRESH_GAUSSIAN_C: 阈值是邻域值减去常数C的高斯加权和。
# blockSize决定了邻域的大小（用来计算阈值的区域大小）。要求大于1且为奇数
# C是从邻域像素的均值或加权和中减去的常数
def threshold_adaptive(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 中值滤波
    img = cv.medianBlur(img, 5)

    # 全局阈值化
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    # Block size 为 11,  C 值为 2
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    titles = ['Original Image', 'Global Threshold (v = 127)', 'Adaptive Mean Threshold', 'Adaptive Gaussian Threshold']
    images = [img, th1, th2, th3]
 
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


# 自定义阈值化
def threshold_custom(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 求整个灰度图像的平均值
    h, w = gray.shape[:2]   # 获取宽高
    m = np.reshape(gray, [1, w*h])  # 变为一维数组
    mean = m.sum() / (w*h)
    print("mean:", mean)

    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imshow("threshold_custom", binary)


# 超大图片阈值化，使用自适应阈值化
# 大图片可先分割成小图片，再使用自适应局部阈值化
def big_image_demo_1(image):
    print(image.shape)
    cw = 200
    ch = 200
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("big_image_demo_gray", gray)

    # 将图片分割成 ch * cw 的小图片再阈值化
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:col+cw]
            # 自适应阈值化
            dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 127, 20)
            gray[row:row + ch, col:col + cw] = dst
            print(np.std(dst), np.mean(dst))    # std标准差, mean均值

    cv.imshow("dst", gray)
    # cv.imwrite("img/result_big_image.png", gray)


# 超大图片阈值化。使用全局阈值化
def big_image_demo_2(image):
    print(image.shape)
    cw = 200
    ch = 200
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("big_image_demo_gray", gray)

    # 将图片分割成 ch * cw 的小图片再阈值化
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:col+cw]

            # 利用标准差消除噪点
            if np.std(roi) < 15:
                gray[row:row + ch, col:col + cw] = 255
            else:
                # 全局阈值化
                ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
                gray[row:row + ch, col:col + cw] = dst

    cv.imshow("dst", gray)
    # cv.imwrite("img/result_big_image.png", gray)


def main():
    img = cv.imread("img/02.jpg")
    # cv.imshow("02", img)

    # threshold_demo(img)

    # threshold_simple(img)

    # threshold_adaptive(img)

    # threshold_custom(img)

    src = cv.imread("img/big_image.jpg")
    cv.imshow("big_image", src)
    big_image_demo_1(src)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()