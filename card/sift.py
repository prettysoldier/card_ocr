'''
refer to:
https://www.jb51.net/article/135366.htm
https://blog.csdn.net/zhangziju/article/details/79754652
https://www.jianshu.com/p/ed57ee1056ab

> python card/sift.py data/1.jpg
'''

import cv2
import numpy as np

card_template_image_path="card.jpg"

key_pos={
	'姓':[[45,60],[65,75]],
	'名':[[80,60],[95,75]],
	'性':[[45,107],[63,123]],
	'别':[[78,109],[96,126]],
	'民':[[191,109],[205,125]],
	'族':[[222,109],[238,126]],
	'出':[[46,156],[62,173]],
	'生':[[78,158],[95,174]],
	'年':[[189,157],[204,173]],
	'月':[[253,156],[264,172]],
	'日':[[310,155],[325,175]],
	'住':[[45,210],[65,225]],
	'址':[[79,208],[96,225]],
	'公':[[47,335],[65,354]],
	'民':[[70,335],[87,350]],
	'身':[[91,335],[105,350]],
	'份':[[111,335],[151,350]],
	'证':[[132,335],[170,350]],
	'号':[[156,335],[190,350]],
	'左上':[[0,0],[20,20]],
	'右上':[[615,0],[635,20]],
	'左下':[[0,379],[20,399]],
	'右上':[[615,379],[635,399]]
}

def filter(p):
	for i, (k, v) in enumerate(key_pos.items()):
		if v[0][0] < p[0] and p[0] < v[1][0] and \
		   v[0][1] < p[1] and p[1] < v[1][1]:
			print("关键点匹配：",k)
			return True
	return False

def parse(dst_image_path):
	sift = cv2.xfeatures2d.SIFT_create()
	# sift = cv2.ORB()

	# 模板图片
	img_tpl = cv2.imread(card_template_image_path)
	gray_tpl= cv2.cvtColor(img_tpl,cv2.COLOR_BGR2GRAY)
	kp_tpl, des_tpl = sift.detectAndCompute(gray_tpl, None)  # des是描述子


	# 目标图片
	img_dst = cv2.imread(dst_image_path)
	# size = img_tpl.shape[0]/ img_dst.shape[0]
	# img_dst = cv2.resize(img_dst,None,fx=size,fy=size)
	gray_dst= cv2.cvtColor(img_dst,cv2.COLOR_BGR2GRAY)
	kp_dst, des_dst = sift.detectAndCompute(gray_dst, None)  # des是描述子

	# hmerge = np.hstack((gray_tpl, gray_dst))  # 水平拼接
	# cv2.imshow("gray", hmerge)  # 拼接显示为gray
	# cv2.waitKey(0)

	# img1 = cv2.drawKeypoints(gray_tpl, kp_tpl, gray_tpl, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
	# img2 = cv2.drawKeypoints(gray_dst, kp_dst, gray_dst, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
	# hmerge = np.hstack((img1, img2))
	# cv2.imshow("point", hmerge)
	# cv2.waitKey(0)

	# BFMatcher解决匹配
	# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# matches = bf.match(des_tpl, des_dst)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des_tpl, des_dst, k=2)
	# matches = sorted(matches, key=lambda x: x[0].distance)
	# matches = matches[:10]
	# 调整ratio
	good = []
	print("一个匹配点数：",len(matches))
	for m, n in matches:
		if m.distance < 0.75 * n.distance:
			i = m.queryIdx
			pt = kp_tpl[i].pt
			if filter(pt):
				good.append([m])

	img3 = cv2.drawMatchesKnn(img_tpl, kp_tpl, img_dst, kp_dst, good, None, flags=2)
	cv2.imshow("BFmatch", img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__=="__main__":
	import sys,os

	imagePath =  sys.argv[1]
	if not os.path.exists(imagePath):print("图片不存在")

	parse(imagePath)

