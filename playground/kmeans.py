import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans, AffinityPropagation


def extract_vector(path):
    resnet_feature_list = []

    for im in glob.glob(path):

        im = cv2.imread(im)
        # print(im)
        im = cv2.resize(im,(224,224))
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(im, None)
        # img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        # resnet_feature = my_new_model.predict(img)
        resnet_feature_np = np.array(des)
        resnet_feature_list.append(resnet_feature_np.flatten())

    return np.array(resnet_feature_list)


array = extract_vector("../images/train/rose.jpg")
print(array)


af = AffinityPropagation(affinity='precomputed').fit(array)
# kmeans = KMeans(n_clusters=2, random_state=0).fit(array)
print(af.labels_)