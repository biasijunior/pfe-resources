import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import sys
sys.path.append('../../')
from functions import functions as fn
import time

algo_start_time = time.time()
img_url = '../../images/testBooks/condame/condame_45.png'
img1 = cv2.imread(img_url, 0) 
print (img1)
# img_url.rsplit('/', 1)[1]         # queryImage
img_url = img_url.rsplit('/', 1)[1]
compare_to_image = img_url.rsplit('.', 1)[0]

sift = cv2.SURF_create()
kp_1, desc_1 = sift.detectAndCompute(img1, None)

bf = cv2.BFMatcher()
title = 'surf_BFMatcher() Match for '
fig = plt.figure()
# Load all the images
for p in np.arange(0.4, 1.05, 0.05):
    p = round(p,2)
    percent = []
    image_names = []
    compute_time_arry = []
    time_for_desc = []

    for image_f in glob.iglob("../../images/testBooks/test/images/*"):
        image_to_compare = cv2.imread(image_f, 0)
        img_name = image_f.rsplit('/', 1)[1]
        # Match descriptors.
        begin_time = time.time()
        # 2) Check for similarities between the 2 images
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        time_for_desc.append(time.time() - begin_time)
        print ("-----description----")

        start_time = time.time()
        matches = bf.knnMatch(desc_1, desc_2, k=2)
        good_points = []

        for m, n in matches:
            if m.distance < p*n.distance:
                good_points.append(m)
        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_1)
        
        number_keypoints = max(len(desc_1),len(desc_2))
        percentage_similarity = float(len(good_points)) / number_keypoints * 100
        total_time = time.time() - start_time

        print("Title: " + img_name)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("Similarity: " + str(int(percentage_similarity)) + " % \n")

        image_names.append(img_name)
        percent.append(int(percentage_similarity))
        compute_time_arry.append(total_time)
        # print type(percent)
        plot_zip = sorted(zip(image_names, percent ,compute_time_arry,time_for_desc),key=lambda pair: pair[1], reverse=True)
        image_names, percent, compute_time_arry, time_for_desc = [list(tup) for tup in zip(*plot_zip)]

        save_zip = zip(image_names,percent, compute_time_arry,time_for_desc)
   
    fn.save_percentage_to_file('knnMatch/surf_10n_0_00_correction_flann.csv', save_zip)
    X = image_names[:6]
    Y = percent[:6]
    plt.plot(X, Y, label=p)
    plt.legend()
    print (image_names[:8], percent[:8],p)
    print ("---------------------------------------------------------------------------------")
print("The total execution time is :  %s seconds" % (time.time() - algo_start_time)) 
plt.xlabel('images')
plt.xticks(rotation=30)
plt.ylabel('percent similarity')   

plt.show()
fig.savefig(title+compare_to_image)


