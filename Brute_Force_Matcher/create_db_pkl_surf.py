import time
import numpy as np
import glob
import cv2
import csv
import pickle as pickle
import sys
# import _pickle as pickle
# Sift and Flann
print("sart")
sift = cv2.xfeatures2d.SURF_create(1000)
bf = cv2.BFMatcher()

# Load all the images
all_images_to_compare = []
titles = []
i = 1

print("loading images \r")

for f in glob.iglob("../images/train/*"):
    imag = cv2.imread(f)
    image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    titles.append(f.rsplit('/', 1)[1])
    print (i, sep=' ', end='', flush=True)
    sys.stdout.flush()
    all_images_to_compare.append(image)
    i += 1

array_des = []
j = 1
print('computing descriptors in percentage(%) ...')
for image_to_compare, title in zip(all_images_to_compare, titles):
        # get key points and descriptors
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        # creating an array of decriptors
        array_des.append(desc_2)
        remaing = j/float(len(titles)) * 100
        print (int(remaing),'|', sep=' ', end='', flush=True)
        j = j + 1
        sys.stdout.flush()
#creating a zipped object of image descriptors with their titles  
zipped_data = zip(array_des, titles)
# creating a .pkl database
# print zipped_data
print('% \n saving to database...')
output = open('../database/book_image_dabase_many_surf.pkl', 'wb')
# writing to a database
pickle.dump(zipped_data, output)
