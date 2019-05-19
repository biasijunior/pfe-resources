import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time
import sys
sys.path.append('..')
import functions.functions as fn

algo_start_time = time.time()

img = cv2.imread("../real_images/ali_crop.jpg")
# img = cv2.imread("../images/testBooks/test/images/original.jpg")
print (img)

original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sift and Flann
# sift = cv2.xfeatures2d.SURF_create(1000)
sift = cv2.ORB_create()
kp_1, desc_1 = sift.detectAndCompute(img, None)
# kp_2, desc_2 = sift.detectAndCompute(img, None)

# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2

search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# im_g = ['original', 'rot_90', 'cropped', 'lumino', 'rot_45', 'other1',
#         'other2', 'other3', 'other4', '0ther5', '0ther6', 'other7', 'other8']

# all_images_to_compare = fn.loadimages("../real_images/test/*")



title = 'orb_flann_Match for '
fig = plt.figure()

for p in np.arange(0.5, 1.05, 0.05):
    p = round(p,2)
    percent = []
    image_names = []
    compute_time_arry = []
    time_for_desc = []
    
    # exit()
    # for image_to_compare, title in all_images_to_compare:
    for image_url in glob.iglob('../real_images/*'):
        image_to_compare = cv2.imread(image_url, 0)
        img_name = image_url.rsplit('/', 1)[1]
        
        print(image_url)

        # 2) Check for similarities between the 2 images
        begin_time = time.time()
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        time_for_desc.append(time.time() - begin_time)
        print ("-----description----")
       
        # print (len(desc_2))
       
        
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
        print (img_name)
        print("Similarity: " + str(percentage_similarity) + "% \n")
        image_names.append(img_name)
        percent.append(int(percentage_similarity))
        compute_time_arry.append(total_time)
        # print type(percent)
        plot_zip = sorted(zip(image_names, percent ,compute_time_arry,time_for_desc),key=lambda pair: pair[1], reverse=True)
        # percent, image = (zip(*plot_zip))
        # image, percent, compute_time_arry, time_for_desc  = (zip(*plot_zip))
        image_names, percent, compute_time_arry, time_for_desc = [list(tup) for tup in zip(*plot_zip)]

        save_zip = zip(image_names,percent, compute_time_arry,time_for_desc)

        # print(save_zip)
        # list(percent)
        # print plot_zip
   
    fn.save_percentage_to_file('flann/surf_10n_0_00_correction_flann.csv', save_zip)
    X = image_names[:8]
    Y = percent[:8]
    plt.plot(X, Y, label=p)
    plt.legend()
    print (image_names[:5], percent[:5],p)
    print ("---------------------------------------------------------------------------------")
print("The total execution time is :  %s seconds" % (time.time() - algo_start_time)) 
plt.xlabel('images')
plt.xticks(rotation=30)
plt.ylabel('percent similarity')   
# plt.cm.gist_ncar(np.random.random())
plt.show()
fig.savefig(title+compare_to_image)
