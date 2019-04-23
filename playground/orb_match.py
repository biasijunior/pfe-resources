import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('../images/testBooks/arabic/arabic_cropped.jpg', 0)          # queryImage
img2 = cv.imread('../images/train/arabic.jpg', 0)  # trainImage
# Initiate ORB detector
# orb = cv.xfeatures2d.SURF_create()
# orb = cv.xfeatures2d.SIFT_create()
# orb = cv.ORB_create()
orb = cv.AKAZE_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# create BFMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf = cv.BFMatcher()

# Match descriptors.
matches = bf.match(des1, des2)

# print matches.shape
# Sort them in the order of their distance.

matches = sorted(matches, key=lambda x: x.distance)

for i in range(0,10):
    # print matches[i].trainIdx
    print matches[i].distance
# Draw first 10 matches.

# img3 = cv.drawMatches(img1, kp1, img2, kp2,matches[:100],None, flags=2)
# plt.imshow(img3), plt.show()
