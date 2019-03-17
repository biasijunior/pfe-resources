import numpy as np
import cv2
from matplotlib import pyplot as plt
import cPickle as pickle

img = cv2.imread('./images/home/home1.jpg',0)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

print(des)
to_write = np.array(des)

output = open('biasi.pkl', 'wb')
pickle.dump(to_write, output)

# draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(img, kp, None,color=(0, 255, 0),flags=0)
# plt.imshow(img2), plt.show()
