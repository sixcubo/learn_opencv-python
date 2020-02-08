# 模板匹配，就是在整个图像区域发现与给定子图像匹配的小块区域，
# 需要模板图像T和待检测图像-源图像S
# 工作方法：在待检测的图像上，从左到右，从上到下计算模板图像与重叠子图像匹配度，
# 匹配度越大，两者相同的可能性越大。
# 作用有局限性，必须在指定的环境下，才能匹配成功，是受到很多因素的影响，所以有一定的适应性

# opencv 有六种模板匹配的算法
# enum { TM_SQDIFF=0, TM_SQDIFF_NORMED=1, TM_CCORR=2, TM_CCORR_NORMED=3, TM_CCOEFF=4, TM_CCOEFF_NORMED=5 };
# M_SQDIFF，TM_SQDIFF_NORMED匹配数值越低表示匹配效果越好，其它四种反之。
# TM_SQDIFF_NORMED，TM_CCORR_NORMED，TM_CCOEFF_NORMED是标准化的匹配，得到的最大值，最小值范围在0~1之间，其它则需要自己对结果矩阵归一化。归一化函数normalize()
# 不同的方法会得到差异很大的结果，可以通过测试选择最合适的方法。


import cv2 as cv
import numpy as np

def template_demo():

    tpl = cv.imread("img/rabbit.jpg")
    target = cv.imread("img/CrystalLiu22.jpg")

    cv.imshow("template", tpl)
    cv.imshow("target", target)

    # 平方差匹配，相关性匹配，相关性系数匹配的标准化的匹配方法，
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]  

    th, tw = tpl.shape[:2]  # 获取模板宽高

    # 使用不同模板匹配方法
    for md in methods:
        print(md)

        result = cv.matchTemplate(target, tpl, md)  # 得到匹配结果
        print(result.shape)

        # minMaxLoc函数 查找全局最小和最大稀疏数组元素并返回其值及其位置
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        # print(min_val, min_loc)

        # cv.TM_SQDIFF_NORMED最小时最相似，其他最大时最相似
        # tl(top-left)为左上角坐标
        if md == cv.TM_SQDIFF_NORMED:  
            tl = min_loc
        else:
            tl = max_loc

        # br(bottom-right)为右下角坐标
        br = (tl[0] + tw, tl[1] + th)

        # 画出矩形
        cv.rectangle(target, tl, br, (0, 0, 255), 2)

        cv.imshow("match-"+np.str(md), target)


template_demo()

cv.waitKey(0) # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
cv.destroyAllWindows()  # 关闭所有窗口