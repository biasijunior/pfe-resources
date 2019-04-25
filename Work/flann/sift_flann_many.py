import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time
import sys
sys.path.append('..')
import functions.functions as fn

start_time = time.time()

img = cv2.imread("../images/test/original_book.jpg")
original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sift and Flann
sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()


flann = cv2.FlannBasedMatcher(index_params, search_params)
# flann = cv2.BFMatcher()
# Load all the images
all_images_to_compare = []
titles = []
for f in glob.iglob("../images/books/test/*"):
    imag = cv2.imread(f)
    image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    titles.append(f)
    all_images_to_compare.append(image)

init_start_time = time.time()
percent = []
image = []
compute_time_arry = []
all_images_to_compare = fn.loadimages("../images/train/*")

for image_to_compare, title in all_images_to_compare:
    start_time = time.time()
    # 1) Check if 2 images are equals
    # if original.shape == image_to_compare.shape:
    #     print("The images have same size and channels")
    # difference = cv2.subtract(original, image_to_compare)
    # b, g, r = cv2.split(difference)

    # if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    #         print("Similarity: 100% (equal size and channels)")
    #         break

    # 2) Check for similarities between the 2 images
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_1)

    print("Title: " + title)
    percentage_similarity = float(len(good_points)) / number_keypoints * 100
    total_time = time.time() - start_time

    print("Similarity: " + str(int(percentage_similarity)) + " %\n")
    print("--- %s seconds ---" % (time.time() - start_time))
    percent.append(str(int(percentage_similarity)))
    image.append(title)
    compute_time_arry.append(total_time)

print("--- total %s seconds ---" % (time.time() - init_start_time))

    
    # img3 = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None, flags=2)

    # plt.imshow(img3,), plt.show()
zipped = zip(image, percent, compute_time_arry)
fn.save_stats_to_file('sift_flann_results_stats.csv', zipped)
