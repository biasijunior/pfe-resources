import numpy as np
import glob
import cv2
from matplotlib import pyplot as plt
import csv
import basehash
import base64
import cPickle as pickle
# import pandas as pd
import array
# import savetxt

hash_fn = basehash.base36()  # you can initialize a 36, 52, 56, 58, 62 and 94 base fn
  # returns 'M8YZRZ'
unhashed = hash_fn.unhash('M8YZRZ')
print unhashed

import time

start_time = time.time()

# Sift and Flann
sift = cv2.ORB_create()
bf = cv2.BFMatcher()

# Load all the images
all_images_to_compare = []
titles = []
for f in glob.iglob("./images/books/test/*"):
    imag = cv2.imread(f)
    image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    titles.append(f)
    all_images_to_compare.append(image)

init_start_time = time.time()
fieldnames = ['names', 'descriptors']
array_des = []
with open('images.csv', 'wb') as csvfile:
    writer = csv.DictWriter(csvfile,delimiter='|', fieldnames=fieldnames)
    writer.writeheader()
    i = 0
    for image_to_compare, title in zip(all_images_to_compare, titles):
        start_time = time.time()
    
    # 2) Check for similarities between the 2 images
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        # zip1 = zip(desc_2)
        array_des.append(desc_2)
        # desc_2 = np.compress(desc_2,kp_2)
        # hash_value = hash_fn.hash(int(str(desc_2)))
        # print desc_2
        # desc_1 = str(desc_2)
        # print("to string")
        # print desc_1
        # print(type(desc_1))
        # print("back to array")
        # desc_1 = np.array(desc_1)
        # print desc_1
        # print(type(desc_1))

    
        # matches = bf.match(desc_1, desc_2)
        # print (type(desc_2))
        # desc_2 = np.array(desc_2)
        # print (type(desc_2))
        print("biasi")
        # pickle_bytes = pickle.dumps(desc_2)            # unsafe to write
        # b64_bytes = base64.b64encode(pickle_bytes)  # safe to write but still bytes
        # b64_str = b64_bytes.decode('utf8')          # safe and in utf8
        # wr.writerow(['col1', 'col2', b64_str])
        # writer.writerow({'names': desc_1, 'descriptors': title})
        writer.writerow({'names': desc_2, 'descriptors': title})
        # writer.writerow(desc_1)
# a = np.array([1,2,3,4])
# b = np.array([5,6,7,8])
img_data = zip(array_des,titles)
output = open('biasii.pkl', 'wb')
# data = zip(obj_1, obj_2)

pickle.dump(img_data, output)
# df = pd.DataFrame({"name1" : a, "name2" : b})
# df.to_csv("submission2.csv", index=False)       
# df = pd.DataFrame({"name1" : a, "name2" : b})
