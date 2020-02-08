import cv2 as cv
import numpy as np


# 图片人脸检测
def face_detection_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)

    # 级联分类器
    face_detector = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_alt_tree.xml")

    # detectMultiScale(image, scaleFactor=None, minNeighbors=None, flags=None, minSize=None, maxSize=None) -> objects
    # . @param image 包含被检测对象的CV_8U类型的矩阵
    # . @param objects 矩形的向量，其中每个矩形都包含检测到的对象。矩形可能部分在原始图像之外
    # . @param scaleFactor 指定图像大小的比例系数
    # . @param minNeighbors 指定每个候选矩形最少应该有多少个相邻的矩形
    # . @param flags Parameter with the same meaning for an old cascade as in the function . cvHaarDetectObjects. It is not used for a new cascade. 
    # . @param minSize Minimum possible object size. Objects smaller than that are ignored. 
    # . @param maxSize Maximum possible object size. Objects larger than that are ignored. If maxSize == minSize model is evaluated on single scale. .
    faces = face_detector.detectMultiScale(gray, 1.02, 2) 
    print(faces)

    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow("result image", image)


# 视频人脸检测
def face_detection_video():
    # 参数为0，使用设备摄像头
    capture = cv.VideoCapture(0)

    # 函数namedWindow可以创建用于images和trackbars的占位窗口，后面使用imshow时会直接使用参数winname相同的窗口
    cv.namedWindow("result image", cv.WINDOW_AUTOSIZE)

    while True:
        # 检测
        ret, frame = capture.read() # 读取帧

        # cv.imshow("1", frame)
        frame = cv.flip(frame, 1)   # 水平翻转

        face_detection_image(frame) # 人脸检测
        c = cv.waitKey(10)     

        # 按下“esc”退出
        if c == 27:
            break



def main():

    # 图片检测
    src = cv.imread("img/CrystalLiu1.jpg")
    cv.imshow("input image", src)
    face_detection_image(src)

    # # 视频检测
    # # 别忘了把刘海撩起来
    # face_detection_video()

    cv.waitKey(0)
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()