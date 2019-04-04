import pprint
import cPickle as pickle
# import _pickle as pickle
import cv2
import time
import csv
import datetime

now = datetime.datetime.now()

# img = cv2.imread('./images/home/home1.jpg', 0)
print('loading test image...')
img = cv2.imread('../images/testBooks/100_samples/1.jpg', 0)
# print img
# Initiate STAR detector
orb = cv2.ORB_create()
# orb = cv2.xfeatures2d.SIFT_create()

# find the keypoints with ORB
# kp = orb.detect(img, None)

# compute the descriptors with ORB
print('extracting descriptors from a test image')
kp, des = orb.detectAndCompute(img, None)
bf = cv2.BFMatcher()

# open a database file
pkl_file = open('../database/book_image_dabase_many.pkl', 'rb')
# get zipped object
print('reading descriptors from a file')
zipped_obj = pickle.load(pkl_file)
pkl_file.close()
print('comparing...')
percent = []
image = []
for image_descriptor, image_name in zipped_obj:
    start_time = time.time()
    matches = bf.knnMatch(des, image_descriptor, k=2)
  
    good_points = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(image_descriptor) <= len(des):
        number_keypoints = len(image_descriptor)
    else:
        number_keypoints = len(des)

    print("Title: " + image_name)
    percentage_similarity = float(len(good_points)) / number_keypoints * 100
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Similarity: " + str((percentage_similarity)) + " %\n")
    percent.append(str(int(percentage_similarity)))
    image.append(image_name)


    # pprint.pprint(data1)
zipped = sorted(zip(percent, image), key=lambda pair: pair[0], reverse= True)
# zipped = sorted(zipped, key = lambda x: x[0])

print('writing results to a file...')
with open('../database/Results_Comparison/sift.csv', 'a') as csvfile:
    fieldnames = ['similarity(%)', 'image name', 'time and date']

    # spamwriter = csv.writer(csvfile, delimiter=' ',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
    writer.writeheader()
    for percentage, name_image in zipped:
        writer.writerow(
            {'similarity(%)': percentage + " %", 'image name': name_image, 'time and date': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
        # writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
        # writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})
print('finished writing to a file')
print('Done!!!')

    
