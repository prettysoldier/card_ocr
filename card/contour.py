import matplotlib
import cv2
import numpy as np
import logging

'''
    这种方法是膨胀，找轮廓，最大的四边形就是！
'''

matplotlib.use('TkAgg')
logger = logging.getLogger("身份证图片识别")

def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])

# def findTextRegion(img):
#     wordInfo = {}
#     height = img.shape[0]
#     width = img.shape[1]
#     # 姓名
#     name_img = img[int(height*0.11):int(height*0.24), int(width*0.18):int(width*0.4)]
#     cv2.imwrite("debug/name.jpg", name_img)
#     # 性别
#     sex_img = img[int(height * 0.25):int(height * 0.35),int(width * 0.18):int(width * 0.25)]
#     cv2.imwrite("debug/sex.jpg", sex_img)
#     # 民族
#     nation_img = img[int(height * 0.24):int(height * 0.34),int(width * 0.39):int(width * 0.44)]
#     cv2.imwrite("debug/nation.jpg", nation_img)
#     # 生日
#     birthday_img = img[int(height * 0.35):int(height * 0.48), int(width * 0.18):int(width * 0.61)]
#     cv2.imwrite("debug/birthday.jpg", birthday_img)
#     # 地址
#     address_img_1 = img[int(height * 0.47):int(height * 0.58), int(width * 0.17):int(width * 0.63)]
#     address_img_2 = img[int(height * 0.59):int(height * 0.68), int(width * 0.17):int(width * 0.63)]
#     cv2.imwrite("debug/address1.jpg", address_img_1)
#     cv2.imwrite("debug/address2.jpg", address_img_2)
#     # 身份证号
#     idcard_img = img[int(height * 0.8):int(height * 0.91), int(width * 0.34):int(width * 0.93)]
#     cv2.imwrite("debug/idcard.jpg", idcard_img)
#     return wordInfo


def filter_gaussian(img):
    # 原来内核大小只支持奇数
    # k = random.choice([(3, 1), (1, 3), (3, 3)])
    blur = cv2.GaussianBlur(img, (5,5), 0)  # （5,5）表示的是卷积模板的大小，0表示的是沿x与y方向上的标准差
    logger.info("高斯模糊")
    return blur


def detect(img):

    #1. 高斯模糊
    blur = filter_gaussian(img)
    cv2.imwrite("debug/blur.jpg", blur)
    logger.info("已生成模糊图【%s】", "debug/blur.jpg")


    #2. 转化成灰度图
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug/gray.jpg", gray)
    logger.info("已生成灰度图【%s】", "debug/gray.jpg")


    # # 3. 自适应二值化方法
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
    _,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("debug/binary.jpg", binary)
    logger.info("已生成二值化图【%s】", "debug/binary.jpg")
    #
    # # 4. 开运算，消除小噪点
    kernel = np.ones((10,10), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imwrite("debug/opening.jpg", binary)
    logger.info("已生成开运算【%s】", "debug/opening.jpg")

    # 膨胀
    kernel = np.ones((30,30), np.uint8)
    dilation = cv2.dilate(opening, kernel)
    cv2.imwrite("debug/dilation.jpg", dilation)
    logger.info("已生成膨胀【%s】", "debug/dilation.jpg")

    # 降噪
    # noise = opencvUtil.noise_remove_cv2(binary,2)
    # cv2.imwrite("debug/noise.jpg", noise)

    # 5. canny边缘检测
    # edged = cv2.Canny(binary, 10, 100)
    # edged = cv2.Canny(gray, 50, 150, apertureSize=3)# 50和150(3*50)是经典的参数大小
    # cv2.imwrite("debug/edge.jpg", edged)
    # logger.info("已生成边缘图【%s】", "debug/edge.jpg")

    # https://blog.csdn.net/sunny2038/article/details/9170013
    # img = dilation
    # x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    # y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # absX = cv2.convertScaleAbs(x)  # 转回uint8
    # absY = cv2.convertScaleAbs(y)
    # edged = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # cv2.imwrite('debug/edged.jpg', edged)
    # logger.info("已生成边缘图【%s】", "debug/edge.jpg")

    # 函数cv2.findContours()三个参数：
    #   第一个是输入图像，第二个是轮廓检索模式，第三个是轮廓近似方法。
    # 返回值有三个：
    #   第一个是图像，第二个是轮廓，第三个是（轮廓的）层析结构。
    #   轮廓（第二个返回值）是一个Python列表，其中储存这图像中所有轮廓。
    #   每一个轮廓都是一个Numpy数组，包含对象边界点（x，y）的坐标。
    cnts_img,cnts,_ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite("debug/contours.jpg", cnts_img)
    logger.info("已生成轮廓图【%s】", "debug/contours.jpg")

    docCnt = None

    # 确保至少有一个轮廓被找到
    if len(cnts) == 0: return None

    # 将轮廓按大小降序排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts = cnts[:10]

    # 对排序后的轮廓循环处理
    poly_img = img.copy()
    for c in cnts:
        logger.debug("轮廓中的点%d个",len(c))
        # 获取近似的轮廓
        peri = cv2.arcLength(c, True)

        # opencv中对指定的点集进行多边形逼近的函数
        # arg1:输入的点集 arg2:指定的精度,也即是原始曲线与近似曲线之间的最大距离  arg3:若为true，则说明近似曲线是闭合的；反之，若为false，则断开
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        logger.debug("由此轮廓围成的多边形%d条边",len(approx))
        cv2.polylines(poly_img, [approx], True, (0, 0, 255), 2)

        # # 如果近似轮廓有四个顶点，那么就认为找到了
        # if len(approx) == 4:
        #     docCnt = approx
        #     break
    else:
        logger.warning("没有轮廓被找到")


    # for i in docCnt:
    #     # circle函数为在图像上作图，新建了一个图像用来演示四角选取
    #     cv2.circle(poly_img, (i[0][0], i[0][1]), 3, (0, 0,255), -1)

    cv2.imwrite("debug/poly.jpg", poly_img)
    logger.info("已生成轮廓多边形图【%s】", "debug/poly.jpg")



    # paper = four_point_transform(image, docCnt.reshape(4, 2))
    # 5.根据4个角的坐标值裁剪图片
    # warped = four_point_transform(gray, docCnt.reshape(4, 2))

    # warped = opencvUtil.grayImg(warped)
    # warped = opencvUtil.binary(warped,120)
    # cv2.imwrite("debug/warped.jpg", warped)

    # # 7. 划分文字区域
    # wordInfo = findTextRegion(warped)
    # return wordInfo


# def four_point_transform(image, docCnt):
#     # 自定义
#     # x1,x2 = max(docCnt[0][0],docCnt[1][0]),min(docCnt[2][0],docCnt[3][0])
#     # y1,y2 = max(docCnt[0][1],docCnt[3][1]),min(docCnt[1][1],docCnt[2][1])
#     # cut_img = image[y1:y2, x1:x2]
#     # opencv
#     x, y, w, h = cv2.boundingRect(docCnt)
#     cut_img = image[y:y + h, x:x + w]
#     return cut_img

# 4点透射变换
def four_point_transform(image, docCnt):
    # 自定义
    # x1,x2 = max(docCnt[0][0],docCnt[1][0]),min(docCnt[2][0],docCnt[3][0])
    # y1,y2 = max(docCnt[0][1],docCnt[3][1]),min(docCnt[1][1],docCnt[2][1])
    # cut_img = image[y1:y2, x1:x2]
    # opencv
    # 原图
    src = np.array([[docCnt[0][0],docCnt[0][1]],[docCnt[3][0],docCnt[3][1]],[docCnt[1][0],docCnt[1][1]],[docCnt[2][0],docCnt[2][1]]],np.float32)
    # 高和宽
    h,w = image.shape[:2]
    # 目标图
    dst = np.array([[0,0],[w,0],[0,h],[w,h]],np.float32)
    P = cv2.getPerspectiveTransform(src, dst)  # 计算投影矩阵
    r = cv2.warpPerspective(img, P, (w, h), borderValue=125)
    return r


# 根据坐标和备注生成wordinfo对象
def getInfo(x,y,w,h,text):
    word_info = {}
    word_info['word'] = text
    pos = []
    pos1 = {}
    pos1['x'] = x
    pos1['y'] = y
    pos2 = {}
    pos2['x'] = x + w
    pos2['y'] = y
    pos3 = {}
    pos3['x'] = x + w
    pos3['y'] = y + h
    pos4 = {}
    pos4['x'] = x
    pos4['y'] = y + h
    pos.append(pos1)
    pos.append(pos2)
    pos.append(pos3)
    pos.append(pos4)
    word_info['pos'] = pos
    return word_info

def gjj_start(img):
    wordInfo = detect(img)
    return wordInfo

if __name__ == '__main__':
    init_logger()
    import sys

    img = cv2.imread(sys.argv[1])
    wordInfo = detect(img)
    logger.info("识别完成")