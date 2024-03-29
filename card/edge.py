import cv2, numpy as np
import matplotlib.pyplot as plt
import os

'''
1.膨胀
2.二值
3.边缘滤波
4.查找轮廓
5.包裹矩形
'''


def imshow(title, img):
    # 中文乱码问题解决不了，搜了所有的帖子，都无解，放弃，采用print
    # title = title.encode("gbk").decode(errors="ignore")
    # title = "我靠".encode("gbk").decode(errors="ignore")
    print(title)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ocr(img_full_path):
    _, img_name = os.path.split(img_full_path)
    img_name = img_full_path
    org_img = img = cv2.imread(img_name, 0)  # 直接读为灰度图像

    print('读入{}'.format(img_name))
    imshow('原图：{}'.format(img_name), org_img)

    # var kernal = Cv.CreateStructuringElementEx(5, 2, 1, 1, ElementShape.Rect);
    #             Cv.Erode(gray, gray, kernal, 2);
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    imshow('debug/{}_膨胀.jpg'.format(img_name), img)

    # 简单滤波
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    imshow('debug/{}_二值-127.jpg'.format(img_name), th1)

    # Otsu 滤波
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(th2)
    imshow('debug/{}_二值-OTSU.jpg'.format(img_name), th2)

    # https://blog.csdn.net/sunny2038/article/details/9170013
    # img = th2
    # x = cv2.Sobel(img,cv2.CV_16S,1,0)
    # y = cv2.Sobel(img,cv2.CV_16S,0,1)
    # absX = cv2.convertScaleAbs(x)# 转回uint8
    # absY = cv2.convertScaleAbs(y)
    # dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    # imshow('debug/{}_X-Sebel滤波.jpg'.format(img_name),absX)
    # imshow('debug/{}_Y-Sebel滤波.jpg'.format(img_name),absY)
    # imshow('debug/{}_Sebel滤波边缘监测.jpg'.format(img_name),dst)

    edge = cv2.Canny(img, 80, 150)
    imshow('Canny边缘', edge)

    # 输出的返回值，image是原图像、contours是图像的轮廓、hier是层次类型
    # https://blog.csdn.net/HuangZhang_123/article/details/80511270
    # 传入的是2值图，输出是轮廓的多边形数组
    img = edge
    black = cv2.cvtColor(np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 将轮廓按大小降序排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    hulls = []
    for cnt in contours:
        # 轮廓周长也被称为弧长。可以使用函数 cv2.arcLength() 计算得到。这个函数的第二参数可以用来指定对象的形状是闭合的（True） ，还是打开的（一条曲线）
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        # 函数approxPolyDP来对指定的点集进行逼近，cnt是图像轮廓，epsilon表示的是精度，越小精度越高，因为表示的意思是是原始曲线与近似曲线之间的最大距离。
        # 第三个函数参数若为true,则说明近似曲线是闭合的，它的首位都是相连，反之，若为false，则断开。
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if cv2.contourArea(cnt) < 1000:
            continue

        print("轮廓点数：", len(approx))

        # convexHull检查一个曲线的凸性缺陷并进行修正，参数cnt是图像轮廓。
        hull = cv2.convexHull(cnt)


        maxs = np.max(hull, axis=0)[0]
        mins = np.min(hull, axis=0)[0]
        hull = np.array([
            [mins[0], mins[1]],
            [maxs[0], mins[1]],
            [maxs[0], maxs[1]],
            [mins[0], maxs[1]]])
        hulls.append(hull)
        cv2.drawContours(edge, [hull], -1, (255,255, 255), 2)


    print("一共%d个轮廓，合格的%d个" %(len(contours),len(hulls)))

    # # 勾画图像原始的轮廓
    # cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
    # # 用多边形勾画轮廓区域
    # cv2.drawContours(black, [approx], -1, (255, 255, 0), 2)
    # 修正凸性缺陷的轮廓区域

    # print(hull.shape) [点数,1,2]
    # print(hulls)

    imshow('debug/{}_合格轮廓.jpg'.format(img_name), edge)

    img = org_img
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for rect in hulls:
        # print(rect.shape)
        cv2.polylines(img, [rect], True, (0, 0, 255))
    imshow('debug/{}_最终探测结果.jpg'.format(img_name), img)

    return img


def main():
    import os

    g = os.walk("data")

    for path, dir_list, file_list in g:
        for file_name in file_list:
            full_path = os.path.join(path, file_name)
            print(full_path)
            out_img = ocr(full_path)
            imshow("output/" + file_name, out_img)


if __name__ == '__main__':
    import sys

    # print(len(sys.argv))
    if len(sys.argv) == 2:
        file_name = sys.argv[1]
        print("处理单个文件：", file_name)
        ocr(file_name)
    else:
        print("处理目录")
        main()
