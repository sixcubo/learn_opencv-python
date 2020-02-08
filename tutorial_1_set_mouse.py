import cv2 as cv
import numpy as np


# 查看所有被支持的鼠标事件。
def search_event():
    events = [i for i in dir(cv) if 'EVENT' in i]
    print(events)


print("----------Hello World!----------")
# 读入图片放进src中
img = cv.imread("CrystalLiu1.jpg")  

# 函数namedWindow可以创建用于images和trackbars的占位窗口，后面使用imshow时会直接使用参数winname相同的窗口
cv.namedWindow("image", cv.WINDOW_AUTOSIZE)  # cv.WINDOW_AUTOSIZE 窗口尺寸自动调整

# search_event()
# 创建图像与窗口并将窗口与回调函数绑定
def draw_circle(event, x, y, flags, param):
    if event==cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img,(x,y),100,(255,255 ,0),2)

cv.setMouseCallback('image', draw_circle)

while True:
    cv.imshow('image', img)
    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()

