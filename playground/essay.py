import os
import datetime
import cv2
import numpy as np
# import ssim.ssimlib as pyssim
from skimage.measure import compare_ssim as ssim
from sklearn.cluster import SpectralClustering, AffinityPropagation
from sklearn import metrics

print(cv2.__version__)
print('biasi')

img1 = cv2.imread('../images/test/original_book.jpg',0)
img2 = cv2.imread('../images/test/original_book.jpg',0)



def get_image_similarity(img1, img2, algorithm='SIFT'):
    #detecting keypoints and descriptors
    sift=cv2.xfeatures2d.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)
    similarity = 0
    for m, n in matches:
        if m.distance < 0.7* n.distance:
            similarity += 1.0
    # Custom normalization for better variance in the similarity matrix
    if similarity == len(matches):
        similarity = 1.0
    elif similarity > 1.0:
        similarity = 1.0 - 1.0/similarity
    elif similarity == 1.0:
        similarity = 0.1
    else:
        similarity = 0.0

    return similarity

def build_similarity_matrix(dir_name, algorithm='SIFT'):
    images = os.listdir(dir_name)
    print images
    num_images = len(images)
    sm = np.zeros(shape=(num_images, num_images), dtype=np.float64)
   
    np.fill_diagonal(sm, 1.0)
    # print sm

    print("Building the similarity matrix using %s algorithm for %d images" %
          (algorithm, num_images))
    start_total = datetime.datetime.now()

    # Traversing the upper triangle only - transposed matrix will be used
    # later for filling the empty cells.
    k = 0
    for i in range(sm.shape[0]):
        for j in range(sm.shape[1]):
            j = j + k
            if i != j and j < sm.shape[1]:
                print images[j]
                sm[i][j] = get_image_similarity(images[i], images[j],algorithm=algorithm)
        k += 1

    # Adding the transposed matrix and subtracting the diagonal to obtain
    # the symmetric similarity matrix
    sm = sm + sm.T - np.diag(sm.diagonal())

    end_total = datetime.datetime.now()
    print("Done - total calculation time: %d seconds" % (end_total - start_total).total_seconds())
    return sm


def get_cluster_metrics(X, labels, labels_true=None):
    metrics_dict = dict()
    metrics_dict['Silhouette coefficient'] = metrics.silhouette_score(X,labels, metric='precomputed')
    if labels_true:
        metrics_dict['Completeness score'] = metrics.completeness_score(labels_true, labels)
        metrics_dict['Homogeneity score'] = metrics.homogeneity_score(labels_true, labels)

    return metrics_dict


def do_cluster(dir_name, algorithm='SIFT', print_metrics=True, labels_true=None):
    matrix = build_similarity_matrix(dir_name, algorithm=algorithm)

    af = AffinityPropagation(affinity='precomputed').fit(matrix)
    af_metrics = get_cluster_metrics(matrix, af.labels_, labels_true)

    print("\nSelected Affinity Propagation for the labeling results")
    return af.labels_

get_image_similarity(img1, img2, algorithm='SIFT')
build_similarity_matrix("../images/train/", algorithm='SIFT')

