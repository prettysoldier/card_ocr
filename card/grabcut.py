# https://www.jianshu.com/p/117f66320589
import numpy as np
import cv2
from matplotlib import pyplot as plt

def gc(image_path):
    img = cv2.imread(image_path)

    h,w,_ = img.shape
    mask = np.zeros(img.shape[:2],np.uint8)# h,w
    bgdModel = np.zeros((1,65),np.float64) # 背景模型，如果为None，函数内部会自动创建一个bgdModel；bgdModel必须是单通道浮点型图像，且行数只能为1，列数只能为13x5
    fgdModel = np.zeros((1,65),np.float64)
    rect = (0,0,h,w)

    cv2.grabCut(img,
                mask=mask,
                rect=rect,
                bgdModel=bgdModel,
                fgdModel=fgdModel,
                iterCount=1,
                mode=cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    plt.imshow(img),plt.colorbar(),plt.show()
if __name__=="__main__":
    import sys,os

    imagePath =  sys.argv[1]
    if not os.path.exists(imagePath):print("图片不存在")

    gc(imagePath)
