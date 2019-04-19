import time
import numpy as np
import glob
import cv2
import csv
import cPickle as pickle
import sys
import create_db_pkl_kaze
# import _pickle as pickle
# Sift and Flann
print("satrt")
sift = cv2.AKAZE_create()
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
    print i,
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
        print int(remaing),'|',
        j = j + 1
        sys.stdout.flush()
#creating a zipped object of image descriptors with their titles  
zipped_data = zip(array_des, titles)
# creating a .pkl database
# print zipped_data
print('% \n saving to database...')
output = open('../database/book_image_dabase_many_akaze.pkl', 'wb')
# writing to a database
pickle.dump(zipped_data, output)

