import time
import numpy as np
import glob
import cv2
import csv
import cPickle as pickle

# Sift and Flann
sift = cv2.ORB_create()
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

# Load all the images
all_images_to_compare = []
titles = []

for f in glob.iglob("./images/train/*"):
    imag = cv2.imread(f)
    image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    titles.append(f)
    print("error")
    all_images_to_compare.append(image)


array_des = []
for image_to_compare, title in zip(all_images_to_compare, titles):
        # get key points and descriptors
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        # creating an array of decriptors
        array_des.append(desc_2)
#creating a zipped object of image descriptors with their titles  
zipped_data = zip(array_des, titles)
# creating a .pkl database
print zipped_data
output = open('./database/book_image_dabase.pkl', 'wb')
# writing to a database
pickle.dump(zipped_data, output)
