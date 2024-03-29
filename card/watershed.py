# https://www.cnblogs.com/zyly/p/9392881.html#_label2
import numpy as np
import cv2

def watershred(image_path):
    # 读入图片
    img = cv2.imread(image_path)

    # 转换为灰度图片
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # canny边缘检测 函数返回一副二值图，其中包含检测出的边缘。
    canny = cv2.Canny(gray_img, 80, 150)
    cv2.imshow('Canny', canny)

    # 寻找图像轮廓 返回修改后的图像 图像的轮廓  以及它们的层次
    canny, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 32位有符号整数类型，
    marks = np.zeros(img.shape[:2], np.int32)
    # findContours检测到的轮廓
    imageContours = np.zeros(img.shape[:2], np.uint8)

    # 轮廓颜色
    compCount = 0
    index = 0
    # 绘制每一个轮廓
    for index in range(len(contours)):
        # 对marks进行标记，对不同区域的轮廓使用不同的亮度绘制，相当于设置注水点，有多少个轮廓，就有多少个轮廓
        # 图像上不同线条的灰度值是不同的，底部略暗，越往上灰度越高
        marks = cv2.drawContours(marks, contours, index, (index, index, index), 1, 8, hierarchy)
        # 绘制轮廓，亮度一样
        imageContours = cv2.drawContours(imageContours, contours, index, (255, 255, 255), 1, 8, hierarchy)

    # 查看 使用线性变换转换输入数组元素成8位无符号整型。
    markerShows = cv2.convertScaleAbs(marks)
    cv2.imshow('markerShows', markerShows)
    # cv2.imshow('imageContours',imageContours)

    # 使用分水岭算法
    marks = cv2.watershed(img, marks)
    afterWatershed = cv2.convertScaleAbs(marks)
    cv2.imshow('afterWatershed', afterWatershed)

    # 生成随机颜色
    colorTab = np.zeros((np.max(marks) + 1, 3))
    # 生成0~255之间的随机数
    for i in range(len(colorTab)):
        aa = np.random.uniform(0, 255)
        bb = np.random.uniform(0, 255)
        cc = np.random.uniform(0, 255)
        colorTab[i] = np.array([aa, bb, cc], np.uint8)

    bgrImage = np.zeros(img.shape, np.uint8)

    # 遍历marks每一个元素值，对每一个区域进行颜色填充
    for i in range(marks.shape[0]):
        for j in range(marks.shape[1]):
            # index值一样的像素表示在一个区域
            index = marks[i][j]
            # 判断是不是区域与区域之间的分界,如果是边界(-1)，则使用白色显示
            if index == -1:
                bgrImage[i][j] = np.array([255, 255, 255])
            else:
                bgrImage[i][j] = colorTab[index]
    cv2.imshow('After ColorFill', bgrImage)

    # 填充后与原始图像融合
    result = cv2.addWeighted(img, 0.6, bgrImage, 0.4, 0)
    cv2.imshow('addWeighted', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    import sys,os

    imagePath =  sys.argv[1]
    if not os.path.exists(imagePath):print("图片不存在")

    watershred(imagePath)

