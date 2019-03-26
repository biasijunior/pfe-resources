import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle as pickle

img = cv2.imread('./images/home/home1.jpg',0)
img2 = img
# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
kp, des1 = orb.compute(img2, kp)

print(type(des))
# to_write = [des, kp]
to_write = {'a': [1, 2.0, 3, 4+6j],
 'b': ('string', u'Unicode string'),
 'c': "{kp}" }
obj_1 = ['test_1', 'come', 'go']
obj_2 = ['names', 'ability', des]
obj_3 = ['test_3', {'ability', 'mobility'}]

output = open('biasi.pkl', 'wb')
data = zip(obj_1,obj_2)

pickle.dump(des, output)
pickle.dump("biasi wiga", output)
pickle.dump(data, output)
# pickle.dump(obj_3, output)


# pickle.dump(des1, output)

# draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(img, kp, None,color=(0, 255, 0),flags=0)
# plt.imshow(img2), plt.show()
