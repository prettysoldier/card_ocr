# refer to: https://www.jishuwen.com/d/2uA2

'''

python card/mser.py data/1.jpg
结论，不行，文字背景的这张1.jpg，就会出现背景中的文字和身份证分不开的情况，
第一步剥离前景看来愈发的重要了，还是尝试用SIFT吧。

'''

import cv2
import numpy as np

def parse(imagePath):
    # 读取图片
    img = cv2.imread(imagePath)

    # 转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 利用Sobel边缘检测生成二值图
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 膨胀、腐蚀
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=2)

    #  查找轮廓和筛选文字区域
    region = []
    image,contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]

        # 计算轮廓面积，并筛选掉面积小的
        area = cv2.contourArea(cnt)
        if (area < 1000):
            continue

        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 1)

        # 找到最小的矩形
        rect = cv2.minAreaRect(cnt)
        print ("rect is: ")
        print (rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 根据文字特征，筛选那些太细的矩形，留下扁的
        if (height > width * 1.3):
            continue

        region.append(box)


    # # 绘制轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    import sys,os

    imagePath =  sys.argv[1]
    if not os.path.exists(imagePath):print("图片不存在")

    parse(imagePath)

