import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time
import sys
sys.path.append('..')
import functions.functions as fn

start_time = time.time()

img = cv2.imread("../../images/train/condame.jpg")
# img2 = cv2.imread("../images/test/original_book.jpg")


original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sift and Flann
sift = cv2.xfeatures2d.SURF_create(500)
# sift = cv2.ORB_create()
kp_1, desc_1 = sift.detectAndCompute(img, None)
# kp_2, desc_2 = sift.detectAndCompute(img, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_LSH,
#                     table_number=6,  # 12
#                     key_size=12,     # 20
#                     multi_probe_level=1)  # 2

search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

im_g = ['original', 'rot_90', 'cropped', 'lumino', 'rot_45','blurred', 'other1',
        'other2', 'other3', 'other4', '0ther5', '0ther6', 'other7', 'other8','other9']

all_images_to_compare = fn.loadimages(
    "../../images/testBooks/test/images/*")
for p in np.arange(0.5, 1.05, 0.05):
    p = round(p,2)
    percent = []
    image = []
    compute_time_arry = []
    time_for_desc = []
    for image_to_compare, title in all_images_to_compare:

        # 2) Check for similarities between the 2 images
        begin_time = time.time()
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        time_for_desc.append(time.time() - begin_time)
        print len(desc_2)
        print "-----description----"
        start_time = time.time()
        matches = flann.knnMatch(desc_1, desc_2, k=2)
        good_points = []
    
        for m_n in matches:
                if len(m_n) != 2:
                    continue
                (m,n) = m_n
                if m.distance < p*n.distance:
                    good_points.append(m)

        number_keypoints = max(len(desc_1),len(desc_2))
        
        percentage_similarity = float(len(good_points)) / number_keypoints * 100
        total_time = time.time() - start_time
        # print("Title: " + title)
        # print("time desc: %s" %(time.time()-begin_time))
        # print("--- %s seconds ---" % (time.time() - start_time))
        # print("Similarity: " + str(int(percentage_similarity)) + "% \n")

        percent.append(int(percentage_similarity))
        image.append(title)
        compute_time_arry.append(total_time)
        # print type(percent)
        plot_zip = sorted(zip(image, percent ,compute_time_arry,time_for_desc),
                        key=lambda pair: pair[1], reverse=True)
        # percent, image = (zip(*plot_zip))
        image, percent, compute_time_arry, time_for_desc = [list(tup) for tup in zip(*plot_zip)]

        save_zip = zip(im_g,percent, compute_time_arry,time_for_desc)

        
        # list(percent)
        # print plot_zip
   
    fn.save_stats_to_file('flann/surf_10*00_correction_flann.csv', save_zip)
    plt.plot(im_g, percent, label=p)
    plt.legend()
    print image, percent,p
    print "---------------------------------------------------------------------------------"
    
plt.xlabel('images')
# plt.xticks(rotation=80)
plt.ylabel('percent similarity')   
# plt.cm.gist_ncar(np.random.random())
plt.show()
