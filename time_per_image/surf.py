import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

start_time = time.time()

img1 = cv2.imread('./images/books/original.jpg', 0)  # queryImage
img2 = cv2.imread('./images/books/original.jpg', 0)  # trainImage

# Initiate surf detector
surf = cv2.xfeatures2d.SURF_create()

# find the keypoints and descriptors with surf
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

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

print("--- %s seconds ---" % (time.time() - start_time))
# cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:100], None, flags=2)

# plt.imshow(img3), plt.show()
