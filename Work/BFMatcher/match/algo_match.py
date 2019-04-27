import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../..')
# import Work.functions.functions
import functions.functions as fn
"""
SIFT edgeThreshold used was 1.5 with distance equale to or less than 100
    i.e. SIFT_create(edgeThreshold=1.5) , matches[i].distance <= 100

TRY MODIFYING THESE PARAMETRES AND ALSO TRY TO MODIFY THE "crossCheck to true/false"

"""
img_url = '../../../images/train/rose.jpg'
img1 = cv.imread(img_url, 0) 
# img_url.rsplit('/', 1)[1]         # queryImage
img_url = img_url.rsplit('/', 1)[1]
compare_to_image = img_url.rsplit('.', 1)[0]


# img2 = cv.imread('../images/testBooks/arabic/arabic_90.jpg', 0)  # trainImage
# Initiate ORB detector
orb = cv.xfeatures2d.SIFT_create()
# orb = cv.ORB_create()
algo_name = 'sift'
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
# create BFMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
bf = cv.BFMatcher()
title = algo_name + '_BFMatcher() Match for '
fig = plt.figure()

images = fn.loadimages('../../../images/testBooks/test/*')
desc_comp_time = []
image_names = []
matching_time = []
percentage_sim = []
match_distance = []
print "sorting matches..."
j = 0
# print images
for img, img_name in images:
    # Match descriptors.
    
    start_time = time.time()
    kp2, des2 = orb.detectAndCompute(img, None)
    desc_comp_time.append(time.time() - start_time)
    print str(time.time() - start_time) + "seconds"
    print img_name
    extr_match = time.time()
    matches = bf.match(des1, des2)
    matching_time.append(time.time() - extr_match)
    # print matches.shape
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
    x = np.arange(len(matches))
    match_distance.append(good_match)
    plt.plot(x, good_match, label=img_name)
    plt.legend()
    # similarity = (float(len(good_match)) / max(len(des1), len(des2))) * 100

    # print (str(similarity) + "%"+"      " + img_name)
    j+=j

#     percentage_sim.append(similarity)
    image_names.append(img_name)
# zipper = zip(image_names, percentage_sim, matching_time, desc_comp_time)

zipper = zip(image_names,matching_time,desc_comp_time)
# print zipper
# print zipper
fn.save_stats_to_file('match/'+algo_name+'_match_distance_for_'+compare_to_image+'.csv',zipper)

plt.title(title + compare_to_image)
plt.xlabel('number of distance point')
plt.ylabel('distance between images')
plt.show()

fig.savefig(title+compare_to_image)
# figure.suptitle('test title', fontsize=20)
# plt.xlabel('xlabel', fontsize=18)
# plt.ylabel('ylabel', fontsize=16)
# figure.savefig('test.jpg')
