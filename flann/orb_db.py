import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import csv
import functions as fn

import time

start_time = time.time()

# Sift and Flann
sift = cv2.ORB_create()

print('comparing...')
percent = []
image = []
compute_time_arry = []
all_images_to_compare = fn.loadimages("../images/train/*")


init_start_time = time.time()
fieldnames = ['first_name', 'last_name', 'something']
with open('images.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
    writer.writeheader()

    for image_to_compare, title in all_images_to_compare:
        start_time = time.time()
    
    # 2) Check for similarities between the 2 images
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        writer.writerow({'first_name': time, 'last_name': kp_2, 'something': desc_2})

fn.save_stats_to_file('akaze_results_stats.csv',zipped)
        
