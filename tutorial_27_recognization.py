import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract as tess

# 步骤：
# 去除干扰线和点
# 选择结构元素
# 将 numpy ndarray对象 转换为 PIL的Image对象
# 识别和输出


def recognition_demo(image):
    # 二值化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    # 去除干扰
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    open_out = cv.morphologyEx(binary,cv.MORPH_OPEN, kernel=kernel)
    cv.imshow("open_out", open_out)

    cv.bitwise_not(open_out, open_out)
    cv.imshow("open_out", open_out)

    textImage = Image.fromarray(open_out)
    text = tess.image_to_string(textImage)

    print("The result:", text)
    

def main():
    src = cv.imread("img/captcha.jpg")
    cv.imshow("src",src)

    recognition_demo(src)

    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()