import numpy as np
import cv2
from matplotlib import pyplot as plt
import cPickle as pickle
import time

start_time = time.time()


def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id, descriptors[i])
        ++i
        temp_array.append(temp)
    return temp_array


def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1],
                                    _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)


img = cv2.imread('./images/home/home1.jpg')
img2 = cv2.imread('./images/home/home1.jpg')

# Initiate STAR detector
orb = cv2.ORB_create()
flann = cv2.BFMatcher()

kp_1, des = orb.detectAndCompute(img, None)
kp_2, des2 = orb.detectAndCompute(img2, None)

#Store and Retrieve keypoint features
temp_array = []
temp = pickle_keypoints(kp_1, des)
temp_array.append(temp)
temp = pickle_keypoints(kp_2, des2)
temp_array.append(temp)

pickle.dump(temp_array, open("keypoints_database.txt", "wb"))

#Retrieve Keypoint Features
keypoints_database = pickle.load(open("keypoints_database.p", "rb"))
kp1, desc_1 = unpickle_keypoints(keypoints_database[0])
kp2, desc_2 = unpickle_keypoints(keypoints_database[1])



matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []

for m,n in matches:
    if m.distance < 0.7*n.distance:
          good_points.append(m)
number_keypoints = 0
if len(kp1) <= len(kp2):
    number_keypoints = len(kp1)
else:
    number_keypoints = len(kp1)

print(len(good_points))
print("Title: ")
percentage_similarity = float(len(good_points)) / len(kp2) * 100
print("--- %s seconds ---" % (time.time() - start_time))
print("Similarity: " + str(int(percentage_similarity)) + "% \n")
