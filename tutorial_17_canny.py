import cv2 as cv
import numpy as np

# canny运算步骤：5步
# 1. 高斯模糊 - GaussianBlur, 去除噪声
# 2. 灰度转换 - cvtColor
# 3. 计算梯度 - Sobel/Scharr
# 4. 非极大值抑制， 将模糊的边界变得清晰（sharp），消除误检
# 5. 高低阈值输出二值图像

# 非极大值抑制：
# 算法使用一个3×3邻域作用在幅值阵列M[i,j]的所有点上；
# 每一个点上，邻域的中心像素M[i,j]与沿着梯度线的两个元素进行比较，
# 其中梯度线是由邻域的中心点处的扇区值ζ[i,j]给出。
# 如果在邻域中心点处的幅值M[i,j]不比梯度线方向上的两个相邻点幅值大，则M[i,j]赋值为零，否则维持原值；
# 此过程可以把M[i,j]宽屋脊带细化成只有一个像素点宽，即保留屋脊的高度值。

# 高低阈值连接
# 阈值T1<T2, 
# 如果边缘像素的值高于T2，则将其标记为强边缘像素，保留
# 如果边缘像素的值小于T1，则将其抑制
# 如果边缘像素的梯度值小于T2且大于T1，则将其标记为弱边缘像素
# 推荐高低阈值比值为 3:1 或 2:1


def edge_demo(image):
    # 高斯模糊
    blurred = cv.GaussianBlur(image, (3, 3), 0)

    # 灰度转换
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    # 直接使用 gray图像
    # Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None) -> edges
    edge_output = cv.Canny(gray, 50, 150)
    
    # 函数重载，使用梯度图像
    # grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)  # CV_16SC1 16位有符号整型单通道矩阵（S符号整型 U无符号整型 F浮点型 C1单通道）
    # grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    # Canny(dx, dy, threshold1, threshold2[, edges[, L2gradient]]) -> edges
    # edge_output = cv.Canny(grad_x, grad_y, 30, 150)
    
    # 彩色边缘
    bgr_edge = cv.bitwise_and(image, image, mask=edge_output)
    
    cv.imshow("gray", gray)
    cv.imshow("edge", edge_output)
    cv.imshow("bgr_edge", bgr_edge)


def main():
    src = cv.imread("img/Crystal.jpg")
    cv.imshow("demo",src)

    edge_demo(src)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()