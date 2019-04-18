import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import csv

import time

start_time = time.time()

# Sift and Flann
sift = cv2.ORB_create()

# Load all the images
all_images_to_compare = []
titles = []
for f in glob.iglob("./images/books/test/*"):
    imag = cv2.imread(f)
    image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    titles.append(f)
    all_images_to_compare.append(image)

init_start_time = time.time()
fieldnames = ['first_name', 'last_name', 'something']
with open('images.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
    writer.writeheader()

    for image_to_compare, title in zip(all_images_to_compare, titles):
        start_time = time.time()
    
    # 2) Check for similarities between the 2 images
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        writer.writerow({'first_name': time, 'last_name': kp_2, 'something': desc_2})
        
