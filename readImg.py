import csv
import cv2
import time
import base64
import numpy as np
import array

# import pandas as pd

start_time = time.time()


img = cv2.imread('./images/home/home1.jpg', 0)

# Initiate STAR detector
orb = cv2.ORB_create()


kp_1, desc_1 = orb.detectAndCompute(img, None)
kp_2, desc_3 = orb.detectAndCompute(img, None)
bf = cv2.BFMatcher()
fieldnames = ['names', 'descriptors']
with open('images.csv', 'rb') as f:
    # reader = csv.reader(f, delimiter='|')
    reader = csv.DictReader(f, delimiter='|', fieldnames=fieldnames)

    # df = np.read_csv('data.csv')
    # data = df.values
    # print data
    
    i = 0
    for row in reader:
       
        print("biasi")
        # desc_2 = row[1]
        print(row)
        # desc_2 = np.array(desc_2)
        # safe to write but still bytes
        # b64_str = desc_2.decode('utf8')
        # b64_bytes = base64.b64decode(b64_str)
        # print (type(desc_2))
        # print b64_bytes
        desc_2 = row['names']
        print (type(desc_2))
        bf = cv2.BFMatcher()
        # # print row[1]
        # name = row[0]
        # kp_2 = row[1]
        if i == 2:
            # print row[0]
            desc_2 = row['names']
            print (type(desc_2))
            # k=2
            # matches = cv2.DescriptorMatcher.knnMatch(desc_1,desc_2,k=2,mask=None,compactResult=None)
            # matches = bf.match(desc_1, desc_2)
            # matches = bf.knnMatch(desc_2, desc_1, k=2)

            # good_points = []
            # for m, n in matches:
            #     if m.distance < 0.6*n.distance:
            #         good_points.append(m)
            # number_keypoints = 0
            # if len(kp_1) <= len(desc_2):
            #     number_keypoints = len(kp_1)
            # else:
            #     number_keypoints = len(kp_1)

            # print("Title: " + name)
            # percentage_similarity = float(len(good_points)) / number_keypoints * 100
            # print("--- %s seconds ---" % (time.time() - start_time))
            # print("Similarity: " + str(int(percentage_similarity)) + " % \n")
            # if name == 'Biasi':
            #     print row[0]
            #     print('equal')
        i = i + 1
print (desc_2)
# desc_2 = np.asarray(desc_2)


matches = bf.knnMatch(desc_1, desc_2, k=2)
