import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time
import sys
sys.path.append('..')
import functions.functions as fn

algo_start_time = time.time()

img_url = '../real_images/bird_kio.jpeg' # queryImage
img = cv2.imread(img_url, 0) 
# img_url.rsplit('/', 1)[1]         
img_url = img_url.rsplit('/', 1)[1]
compare_to_image = img_url.rsplit('.', 1)[0]

# Sift and Flann
sift = cv2.xfeatures2d.SIFT_create()
algo_name = 'sift'
# sift = cv2.ORB_create()
kp_1, desc_1 = sift.detectAndCompute(img, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

title = algo_name
fig = plt.figure()
for p in np.arange(0.5, 1.05, 0.05):
    p = round(p,2)
    percent = []
    image_names = []
    compute_time_arry = []
    time_for_desc = []
    j = 0
    # exit()
    # for image_to_compare, title in all_images_to_compare:
    for image_url in glob.iglob('../real_images/ali_*'):
        image_to_compare = cv2.imread(image_url, 0)
        img_name = image_url.rsplit('/', 1)[1]
        # 2) Check for similarities between the 2 images
        begin_time = time.time()
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        time_for_desc.append(time.time() - begin_time)
        print ("-----description----")
        j = j + 1
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
        print("Title: " + img_name + "  is number  " + str(j) + "  :::: for p = " + str(p))
        print("Similarity: " + str(percentage_similarity) + "% \n")
        image_names.append(img_name)
        percent.append(int(percentage_similarity))
        compute_time_arry.append(total_time)
        # print type(percent)
        plot_zip = sorted(zip(image_names, percent ,compute_time_arry,time_for_desc),key=lambda pair: pair[1], reverse=True)
        
        image_names, percent, compute_time_arry, time_for_desc = [list(tup) for tup in zip(*plot_zip)]
        save_zip = zip(image_names,percent, compute_time_arry,time_for_desc)
        
    fn.save_percentage_to_file('../database/flann/flann_'+algo_name+'.csv', save_zip)
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
