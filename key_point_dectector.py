
#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('book1b.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sift = cv2.xfeatures2d.SURF_create()
sift = cv2.AKAZE_create()
# sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(gray,kp, None)

# arr = np.array(img)
# cv2.rectangle(img, (384,0),(510,128),(0,255,0),3)
# cv2.imwrite()
cv2.imwrite('keypoint_detector/kaze_keypoints_book1b.png',img)


