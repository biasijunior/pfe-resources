import pprint
import cPickle as pickle
import cv2
import time

start_time = time.time()


img = cv2.imread('./images/home/home1.jpg', 0)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

pkl_file = open('biasii.pkl', 'rb')

# with open('biasi.pkl', 'rb') as f:

#     data1 = pickle.load(f)
#     # data = pickle.loads(f)
#     print data1
#     name = pickle.load(f)
#     print name
#     # print data
#     obj_1 = pickle.load(f)
obj_1 = pickle.load(pkl_file)
i = 0
for image_to_compare, title in obj_1:
# obj_2 = pickle.load(pkl_file)
# obj_3 = pickle.load(pkl_file)

    img = image_to_compare
    print(type(image_to_compare))
# print(obj_2)
# # print(obj_3)
print img
# print name
data1 = img

print(type(data1))

flann = cv2.BFMatcher()
matches = flann.knnMatch(des, data1, k=2)

# pprint.pprint(matches)


good_points = []
for m, n in matches:
    if m.distance < 0.6*n.distance:
        good_points.append(m)
number_keypoints = 0
if len(data1) <= len(des):
    number_keypoints = len(des)
else:
    number_keypoints = len(des)

print("Title: ")
percentage_similarity = float(len(good_points)) / number_keypoints * 100
print("--- %s seconds ---" % (time.time() - start_time))
print("Similarity: " + str(int(percentage_similarity)) + " %\n")


# pprint.pprint(data1)

pkl_file.close()
