

'''

没太仔细研究haar算法，就知道它是检测角的，就考虑是不是可以把阈值啥的，调大点，可以检测出来身份证的圆角，
结果失败，调了调参数，31,29，已经是最大可以可用的了，但是还是不行，检测出来的都是小角，这种大圆角没戏。

'''

import cv2
import numpy as np

def parse(imagePath):
    # 读取图片
    img = cv2.imread(imagePath)

    BlockSize = (7,   9,31)
    Ksize =     (11, 23,29)

    gray = np.float32(img)

    # 转化成灰度图
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    for i, j in zip(BlockSize, Ksize):
        '''
     　　• img - 数据类型为 float32 的输入图像。
     　　• blockSize - 角点检测中要考虑的领域大小。
     　　• ksize - Sobel 求导中使用的窗口大小
     　　• k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06].'''
        dst = cv2.cornerHarris(src=gray, blockSize=i, ksize=j, k=0.06)
        # 变量a的阈值为0.01 * dst.max()，如果dst的图像值大于阈值，那么该图像的像素点设为True，否则为False
        # 将图片每个像素点根据变量a的True和False进行赋值处理，赋值处理是将图像角点勾画出来
        print( dst.max())
        a = dst > 0.01 * dst.max()
        img[a] = [0, 0, 255]
        cv2.imshow('corners_' + str(i) + '_' + str(j), img)
        cv2.waitKey(0)  # 按Esc查看下一张

    cv2.destroyAllWindows()

if __name__=="__main__":
    import sys,os

    imagePath =  sys.argv[1]
    if not os.path.exists(imagePath):print("图片不存在")

    parse(imagePath)

