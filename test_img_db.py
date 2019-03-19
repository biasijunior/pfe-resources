import pprint
import cPickle as pickle
import cv2
import time

start_time = time.time()


# img = cv2.imread('./images/home/home1.jpg', 0)
img = cv2.imread('./images/test/original_book.jpg', 0)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
flann = cv2.BFMatcher()

# open a database file
pkl_file = open('image_dabase.pkl', 'rb')
# get zipped object
zipped_obj = pickle.load(pkl_file)
pkl_file.close()
for image_descriptor, image_name in zipped_obj:
    matches = flann.knnMatch(des, image_descriptor, k=2)
  
    good_points = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(image_descriptor) <= len(des):
        number_keypoints = len(des)
    else:
        number_keypoints = len(des)

    print("Title: " + image_name)
    percentage_similarity = float(len(good_points)) / number_keypoints * 100
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Similarity: " + str(int(percentage_similarity)) + " %\n")


    # pprint.pprint(data1)

    
