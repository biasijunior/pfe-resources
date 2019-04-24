import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import functions as fn


img1 = cv.imread('../images/train/prayer.jpg', 0)          # queryImage
# img2 = cv.imread('../images/testBooks/arabic/arabic_90.jpg', 0)  # trainImage
# Initiate ORB detector
orb = cv.xfeatures2d.SIFT_create(edgeThreshold=1.5)
# orb = cv.ORB_create()

# find the keypoints and descriptors with ORB

kp1, des1 = orb.detectAndCompute(img1, None)


# create BFMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck = False)
images = fn.loadimages('../images/testBooks/test/*')
desc_comp_time = []
image_names = []
matching_time = []
percentage_sim = []
print "sorting matches..."
for img, img_name in images:
    # Match descriptors.
    start_time = time.time()
    kp2, des2 = orb.detectAndCompute(img, None)
    desc_comp_time.append(time.time() - start_time)
    print str(time.time() - start_time) + "seconds"

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
        # print p1
        if p1 <= 100:
            good_match.append(p1)
        
        # print('%.5f' % p1)


    similarity = (float(len(good_match)) / max(len(des1), len(des2))) * 100

    print (str(similarity) + "%"+"      " + img_name)
    percentage_sim.append(similarity)
    image_names.append(img_name)

zipper = zip(image_names,percentage_sim,matching_time,desc_comp_time)

fn.save_stats_to_file('sift_match_100.csv',zipper)
