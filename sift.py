import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('home1.jpg', 0)  # queryImage
img2 = cv2.imread('home3.png', 0)  # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()



# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)


dif = len(des1) - len(des2)
print(len(matches))

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.9 * n.distance:
        # print(m.distance)
        good.append([m])

per = float(len(good))/float(len(des2))

per = per * 100
print(per, 'percent')
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:100], None, flags=2)

plt.imshow(img3), plt.show()
