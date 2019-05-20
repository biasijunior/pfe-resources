import cv2
import numpy as np
import glob
<<<<<<< HEAD
# from matplotlib import pyplot as plt
=======
from matplotlib import pyplot as plt
>>>>>>> 2b6486dfb15b4d8e13b2c293cfc796f4a0132278
import csv
# import winsound

import time
import functions as fn



img = cv2.imread("../images/train/condame.jpg")
original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sift and Flann
sift = cv2.ORB_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)

bf = cv2.BFMatcher()

print('comparing...')
percent = []
image = []
compute_time_arry = []
all_images_to_compare = fn.loadimages("../images/testBooks/test/*")

for image_to_compare, title in all_images_to_compare:
    
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    start_time = time.time()
    matches = bf.knnMatch(desc_1, desc_2, k=2)
     
    good_points = []
    for m, n in matches:
        if m.distance < p*n.distance:
            good_points.append(m)
    
    number_keypoints = max(len(desc_1), len(desc_2))
    percentage_similarity = float(len(good_points)) / number_keypoints * 100
    total_time = time.time() - start_time

    print("Title: " + title)
    print("--- %s seconds ---" % (total_time))
    print("Similarity: " + str((percentage_similarity)) + " %\n")

    percent.append(int(percentage_similarity))
    image.append(title)
    compute_time_arry.append(total_time)
    plt.plot(image,percentage_similarity, label=p)


    # pprint.pprint(data1)
# zipped = sorted(zip(percent, image), key=lambda pair: pair[0], reverse= True)
# zipped = sorted(zipped, key = lambda x: x[0])

# zipped = zip(image,percent,compute_time_arry)

# print('writing results to a file...')

# fn.save_stats_to_file('orb_resultats.csv',zipped)
# winsound.MessageBeep()


    
