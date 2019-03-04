import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('./images/home/home1.jpg', 0)  # queryImage
img2 = cv2.imread('./images/home/home3.png', 0)  # trainImage

# Initiate AKAZE detector
akaz = cv2.AKAZE_create()


# find the keypoints and descriptors with akaz
kp1, des1 = akaz.detectAndCompute(img1, None)
kp2, des2 = akaz.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher() #creates BFMatcher object
matches = bf.knnMatch(des1, des2, k=2)


dif = len(des1) - len(des2)
print(len(matches))

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.9 * n.distance:
        # print(m.distance)
        good.append([m])

percentage_similarity = float(len(good))/float(len(des2)) * 100

print("Similarity: " + str(int(percentage_similarity)) + "% \n")
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:100], None, flags=2)

plt.imshow(img3), plt.show()
