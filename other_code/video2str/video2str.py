'''
---------------------
使用openCV将视频字符化
---------------------
'''

from cv2 import *


def main():

    character_list = list(' .*$#')    # 初始化字符串列表

    video = VideoCapture('news.gif')    # 初始化视频

    active, frame = video.read()    # 获取视频首帧

    txt_list = []   # 初始化输出列表

    n = 0

    while active:
        txt = ''    # 每帧处理为一个字符串

        gray_frame = cvtColor(frame, COLOR_BGR2GRAY)    # 帧转化为灰度图片

        frame_video = resize(gray_frame, (100, 25))      # resize,适应控制台大小

        # 处理帧的每个像素
        for gray_frame_line in frame_video:
            for gray_pixel in gray_frame_line:
                # 每个像素根据灰度对应字符
                txt += character_list[int(gray_pixel /
                                          (256/len(character_list)))]
            txt += '\n'
            n += 1

        txt_list.append(txt)    # 添加到输出列表
        print(n)

        active, frame = video.read()    # 获取视频后续帧

    # 输出
    for str_frame in txt_list:
        waitKey(25)
        print(str_frame)


print(__doc__)
k = input('输入Y/y运行程序，或输入任意键退出：')
if k == 'Y' or k == 'y':
    main()
else:
    print('')
