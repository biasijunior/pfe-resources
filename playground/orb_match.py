import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
img1 = cv.imread('../images/testBooks/arabic/arabic_blurred.jpg', 0)          # queryImage
img2 = cv.imread('../images/train/arabic.jpg', 0)  # trainImage
# img1 = cv.imread('../images/testBooks/arabic/arabic_cropped.jpg', 0)          # queryImage
# img2 = cv.imread('../images/train/condame.jpg', 0)  # trainImage
# img1 = cv.imread('../images/train/arabic.jpg', 0)          # queryImage
# img2 = cv.imread('../images/train/arabic.jpg', 0)  # trainImage
# Initiate ORB detector
# orb = cv.xfeatures2d.SURF_create()
# orb = cv.xfeatures2d.SIFT_create()
# orb = cv.ORB_create()
orb = cv.AKAZE_create()


img1 = cv.imread('../images/train/arabic.jpg', 0)          # queryImage
img2 = cv.imread('../images/testBooks/arabic/arabic.jpg', 0)  # trainImage
# Initiate ORB detector
orb = cv.xfeatures2d.SIFT_create()
# orb = cv.ORB_create()

# find the keypoints and descriptors with ORB

kp1, des1 = orb.detectAndCompute(img1, None)
start_time = time.time()
kp2, des2 = orb.detectAndCompute(img2, None)

print str(time.time() - start_time) + "seconds"

# create BFMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

# Match descriptors.
matches = bf.match(des1, des2)

# print matches.shape
# Sort them in the order of their distance.


# print matches.shape
# Sort them in the order of their distance.


print len(des1)
print "sorting matches..."

matches = sorted(matches, key=lambda x: x.distance)

for i in range(0,100):
    # print matches[i].trainIdx
    print matches[i].distance
# Draw first 10 matches.
print type(matches.distance)

# print matches.distance[10]
# p1 = matches[0].distance
# print p1
similarity = (float(len(matches)) / min(len(des1), len(des2))) * 100


print (str(similarity) + "%")

# for m ,n in matches:
#     print m.distance
good_match=[]
print "//////////-------///////////////----////////////////-----///////////"
for i in range(0,len(des1)):
    p1 = matches[i].distance
    good_match.append(p1)
    print p1
    # if p1 <= 100:
    #     good_match.append(p1)
    
    # print('%.5f' % p1)
# print len(p1)
# plt.plot(['rotated_45', 'rotated_90', 'cropped', 'blurred'], [
#          13.18976329, 2.969968061, 2.905574723, 2.722227371])
x = np.arange(len(des1))
plt.plot(x,good_match)
similarity = (float(len(good_match)) / min(len(des1), len(des2))) * 100

print (str(similarity) + "%")
# plt.show()
    # print p1
# print(p1)

# print('%.5f' % p1)
# img3 = cv.drawMatches(img1, kp1, img2, kp2,matches[:100],None, flags=2)
# plt.imshow(img3), plt.show()
