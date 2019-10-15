# 导入需要的库。
# 参考：https://yq.aliyun.com/articles/547689
import imutils
import numpy as np
import cv2

# 定义 Radon 变换函数，检测范围-90 至 90,间隔为 0.5：

def radon_angle(img, angle_split=5):
    angles_list = list(np.arange(-90., 90. + angle_split,
                                 angle_split))

    # 创建一个列表 angles_map_max，存放各个方向上投影的积分最大
    # 值。我们对每个旋转角度进行计算，获得每个角度下图像的投影，
    # 然后计算当前指定角度投影值积分的最大值。最大积分值对应的角度
    # 即为偏转角度。

    angles_map_max = []
    for current_angle in angles_list:
        rotated_img = imutils.rotate_bound(img, current_angle)
        current_map = np.sum(rotated_img, axis=1)
        angles_map_max.append(np.max(current_map))

    adjust_angle = angles_list[np.argmax(angles_map_max)]

    return adjust_angle


def adjust(img_path):
    img = cv2.imread(img_path)

    degree = radon_angle(img)

    from scipy import ndimage
    rotated = ndimage.rotate(img,degree )
    cv2.imshow('img', rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    import sys,os

    imagePath =  sys.argv[1]
    if not os.path.exists(imagePath):print("图片不存在")

    adjust(imagePath)
