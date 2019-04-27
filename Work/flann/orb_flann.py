import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time
import sys
sys.path.append('..')
import functions.functions as fn

start_time = time.time()

# img = cv2.imread("../../images/train/prayer.jpg")
# img2 = cv2.imread("../images/test/original_book.jpg")


original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sift and Flann
sift = cv2.ORB_create()
kp_1, desc_1 = sift.detectAndCompute(img, None)
kp_2, desc_2 = sift.detectAndCompute(img, None)

FLANN_INDEX_LSH = 6

index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2

search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


percent = []
image = []
compute_time_arry = []
all_images_to_compare = fn.loadimages("../../images/testBooks/test/*")

for image_to_compare, title in all_images_to_compare:

    # 2) Check for similarities between the 2 images
    begin_time = time.time()
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    total_time2 = time.time()-begin_time
    start_time = time.time()
    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []

    for p in np.arange(0.5,1,0.05):

        for m_n in matches:
            if len(m_n) != 2:
                continue
            (m,n) = m_n
            if m.distance < p*n.distance:
                good_points.append(m)

    # for m, n in matches:
    #     if m.distance < 1*n.distance:
    #         good_points.append(m)
    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_1)

    print("Title: " + title)
    percentage_similarity = float(len(good_points)) / number_keypoints * 100
    total_time = time.time() - start_time

    # print("time desc: %s" %(time.time()-begin_time))
    # print("--- %s seconds ---" % (time.time() - start_time))
    print("Similarity: " + str(int(percentage_similarity)) + "% \n")
    percent.append(str(int(percentage_similarity)))
    image.append(title)
    compute_time_arry.append(total_time)
