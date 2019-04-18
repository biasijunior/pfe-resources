import time
import numpy as np
import glob
import cv2
import csv
import cPickle as pickle
import sys
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn import cluster

# from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# import _pickle as pickle
# Sift and Flann
print("satrt")
sift = cv2.xfeatures2d.SIFT_create()
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

array_des = np.array([])
j = 1
print('computing descriptors in percentage(%) ...')
for image_to_compare, title in zip(all_images_to_compare, titles):
        # get key points and descriptors
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        # creating an array of decriptors
        # array_des.append(desc_2)
        array_des = np.append(array_des, desc_2)
        remaing = j/float(len(titles)) * 100
        print int(remaing), '|',
        j = j + 1
        sys.stdout.flush()
#creating a zipped object of image descriptors with their titles
# array_des = np.array(array_des)
print array_des.shape
desc = np.reshape(array_des, (len(array_des)/128, 128))

desc = np.float32(desc)

kmeans = cluster.KMeans(14)
kmeans.fit(desc)

# plt.scatter(desc[:, 0], desc[:, 1], label='True Position')
# img = cv2.drawKeypoints(img, kp_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imwrite('sift_keypoints.jpg', img)
print(kmeans)

# img3 = cv.drawMatches(img, kp,des[:100],None, flags=2)
# imshow(img3), plt.show()
# print(kmeans.cluster_centers_)
# plt.scatter(kmeans.cluster_centers_[:, 1],
#             kmeans.cluster_centers_[:, 1], color='black')
plt.scatter(desc[:, 0], desc[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], color='black')
# plt.show()
# print(kmeans.cluster_centers_)
print(kmeans.labels_)


zipped_data = zip(array_des, titles)
# creating a .pkl database
# print zipped_data
print('% \n saving to database...')
# output = open('../database/book_image_dabase_many_sift.pkl', 'wb')
# writing to a database
# pickle.dump(zipped_data, output)
