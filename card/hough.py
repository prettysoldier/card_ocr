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

# 先通过hough transform检测图片中的图片，计算直线的倾斜角度并实现对图片的旋转
# https://www.cnblogs.com/luofeel/p/9150968.html
import os
import cv2
import math
import numpy as np
from scipy import misc, ndimage


def hough(img):
    img = cv2.GaussianBlur(img, (7,7), 0)  # （5,5）表示的是卷积模板的大小，0表示的是沿x与y方向上的标准差
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None
    edges = cv2.Canny(image=img,
                      threshold1=10,
                      threshold2=100,
                      apertureSize=3,
                      L2gradient=True)

    print("edges.shape:", edges.shape)
    misc.imsave('debug/edges.jpg', edges)

    # 霍夫变换
    minLineLength = 500  # 最小直线长度，太小舍去
    maxLineGap = 1  # 最大线段间隙， 太大舍去
	# 参数详解：https://zhuanlan.zhihu.com/p/34114020
    lines = cv2.HoughLines(
		image=edges,
		rho=1,
		theta=np.pi/180,
		threshold=100,
		lines=minLineLength,
		srn=maxLineGap)

    if lines is None:
        print("无法找到任何直线！")
        return None

    print("lines.shape:",lines.shape)
    lines = lines[:100]
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
			#变换到xy坐标系
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # print(x0, y0, x1, y1, x2, y2)
            cv2.line(edges, (x1, y1), (x2, y2), (255, 255, 255), 3)
    misc.imsave('debug/lines.jpg', edges)
    if x1 == x2 or y1 == y2:
        print("error")
        exit(0)
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    rotate_img = ndimage.rotate(img, rotate_angle)
    misc.imsave('debug/result.jpg', rotate_img)


def ocr(img_full_path):
    _, img_name = os.path.split(img_full_path)
    img_name = img_full_path
    org_img = img = cv2.imread(img_name, 0)  # 直接读为灰度图像

    img = hough(org_img)
    return img


def main():
    import os

    g = os.walk("data")

    for path, dir_list, file_list in g:
        for file_name in file_list:
            full_path = os.path.join(path, file_name)
            print(full_path)
            out_img = ocr(full_path)
            cv2.imwrite("output/" + file_name, out_img)


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
