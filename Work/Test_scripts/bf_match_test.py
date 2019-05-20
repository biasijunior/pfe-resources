import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import sys
import glob
sys.path.append('../')
# import Work.functions.functions
import functions.functions as fn
"""
SIFT edgeThreshold used was 1.5 with distance equale to or less than 100
    i.e. SIFT_create(edgeThreshold=1.5) , matches[i].distance <= 100

TRY MODIFYING THESE PARAMETRES AND ALSO TRY TO MODIFY THE "crossCheck to true/false"

"""
img_url = '../real_images/bird_kio.jpeg'
img1 = cv.imread(img_url, 0) 
print (img1)
# img_url.rsplit('/', 1)[1]         # queryImage
img_url = img_url.rsplit('/', 1)[1]
compare_to_image = img_url.rsplit('.', 1)[0]

# Initiate ORB detector
# orb = cv.xfeatures2d.SIFT_create()
orb = cv.ORB_create()
algo_name = 'orb'
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
# create BFMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
bf = cv.BFMatcher()
title = algo_name + 'match for'
fig = plt.figure()
num_plots = 15
# plt.cm.gist_ncar
# colormap = plt.cm.gist_ncar
# .set_prop_cycle
# plt.gca().set_color_cycle([colormap(i)
#                            for i in np.linspace(0, 0.78, num_plots)])

time_started = time.time()
# images = fn.loadimages('../../../../real_images/*')
desc_comp_time = []
image_names = []
matching_time = []
percentage_sim = []
match_distance = []
print ("sorting matches...")
j = 0

for image_f in glob.iglob('../real_images/test/*'):
        # print path_to_images
    img = cv.imread(image_f, 0)
    img_name = image_f.rsplit('/', 1)[1]
    image_names.append(img_name)
    # Match descriptors.
    j = j + 1
    start_time = time.time()
    kp2, des2 = orb.detectAndCompute(img, None)
    desc_comp_time.append(time.time() - start_time)
    print (str(time.time() - start_time) + "seconds")
    print("Title: " + img_name + "  is number  " + str(j) + "  :::: for p = " + str(p))
    extr_match = time.time()
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    good_match=[]
    # print "//////////-------///////////////----////////////////-----///////////"
    for i in range(0,len(matches)):
        p1 = matches[i].distance
        good_match.append(p1)
        # print p1
        # if p1 <= 100:
        #     good_match.append(p1)
        
        # print('%.5f' % p1)
    matching_time.append(time.time() - extr_match)
    x = np.arange(15)
    match_distance.append(good_match)
    plt.plot(x, good_match[:15], label=img_name)
    plt.legend()
    # similarity = (float(len(good_match)) / max(len(des1), len(des2))) * 100

    # print (str(similarity) + "%"+"      " + img_name)
    j = j + 1

#     percentage_sim.append(similarity)
    image_names.append(img_name)
# zipper = zip(image_names, percentage_sim, matching_time, desc_comp_time)

zipper = zip(image_names,matching_time,desc_comp_time)
# print zipper
# print zipper
fn.save_descriptors_to_file('../database/TestResults/' +algo_name+'_match_test.csv',zipper)
print("finished after:  " + str(time.time()-time_started) + "   :seconds")
plt.title("compared to " + img_url)
plt.xlabel('number of descriptors')
plt.ylabel('distance')
# plt.cm.gist_ncar(np.random.random())
plt.show()

fig.savefig(title+compare_to_image)
