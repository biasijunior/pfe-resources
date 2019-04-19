import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import csv

import time
import functions as fn

start_time = time.time()

img = cv2.imread("../images/test/original_book.jpg")
original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sift and Flann
sift = cv2.ORB_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)

bf = cv2.BFMatcher()

print('comparing...')
percent = []
image = []
compute_time_arry = []
all_images_to_compare = fn.loadimages("../images/train/*")

for image_to_compare, title in all_images_to_compare:
    start_time = time.time()
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    matches = bf.knnMatch(desc_1, desc_2, k=2)
     
    good_points = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)
    number_keypoints = 0
    number_keypoints = 0
    if len(desc_2) <= len(desc_1):
        number_keypoints = len(desc_2)
    else:
        number_keypoints = len(des_1)

    print("Title: " + title)
    percentage_similarity = float(len(good_points)) / number_keypoints * 100
    total_time = time.time() - start_time
    print("--- %s seconds ---" % (total_time))
    print("Similarity: " + str((percentage_similarity)) + " %\n")
    percent.append(str(int(percentage_similarity)))
    image.append(title)
    compute_time_arry.append(total_time)


    # pprint.pprint(data1)
# zipped = sorted(zip(percent, image), key=lambda pair: pair[0], reverse= True)
# zipped = sorted(zipped, key = lambda x: x[0])
zipped = zip(image,percent,compute_time_arry)

print('writing results to a file...')

fn.save_stats_to_file('sift_results.csv',zipped)


    
