import os
import numpy as np
import cv2
# import imgcluster
from matplotlib import pyplot as plt
import save_cluster as clusters
import time
import pickle
import csv
import save_cluster as clusters
DIR_NAME = '../Work/original_images/'
cluster = clusters.read_clusters('orb_cluster_match.pkl')
images = images = os.listdir(DIR_NAME)
num_clusters = len(cluster)
center_index = clusters.read_clusters('orb_centers_match.pkl')

class Matching_Algorithm:
    
    def __init__(self, algorithm_to_use, image_url, bf_or_flann_matcher, get_train_images_url=""):
        algorithm = algorithm_to_use.upper()
        matcher_obj = bf_or_flann_matcher.upper()
        self.image = cv2.imread(image_url, cv2.COLOR_BGR2GRAY)
        # self.matcher_obj = cv2.BFMatcher()
        self.matcher_name = bf_or_flann_matcher
        self.train_images_url = get_train_images_url
        self.get_matcher(matcher_obj)
        self.create_algorithm_obj(algorithm)
        #started here
        self.images = images = os.listdir(DIR_NAME)

        self.center_index = clusters.read_clusters('orb_centers_match.pkl')
        image_descriptors = open('../database/orb_descriptors_dic.pkl', 'rb')
        # get zipped object
        print('reading descriptors from a file')
        self.desc_dict = pickle.load(image_descriptors)

        
    def create_algorithm_obj(self, algorithm):
        try:
            if algorithm == "SIFT":
                self.algorithm = cv2.xfeatures2d.SIFT_create()
            elif algorithm == "SURF":
                self.algorithm = cv2.xfeatures2d.SURF_create()
            elif algorithm == "ORB":
                self.algorithm = cv2.ORB_create()
            elif algorithm == "AKAZE":
                self.algorithm = cv2.AKAZE_create()
            elif algorithm == "KAZE":
                self.algorithm = cv2.KAZE_create()
            else:
                raise Exception("object not created")
            
            print (algorithm + " object successfully created")
        except Exception as error:
            print (repr(error) + "\nError on algorith_name, ENTER a valid algorithm name e.g ORB, SIFT etc")
            exit()


    def get_matcher(self, matcher_obj):
        try:
            if matcher_obj == "BF":
                self.matcher_obj = cv2.BFMatcher()
            elif matcher_obj == "FLANN":
                index_params = dict(algorithm=0, trees=5)
                search_params = dict()
                self.matcher_obj = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                raise Exception("Wrong matcher!")
            print ("You have chosen "+ matcher_obj + "matcher")
        except Exception as error:
            print (repr(error) + "\n enter bf or flann")
            exit()



    def get_keypoint_and_desc(self):
       self.kp_1 , self.desc_1 = self.algorithm.detectAndCompute(self.image, None)
       return self.kp_1, self.desc_1
    
    def get_percent_similarity(self,desc_2):

        matches = self.matcher_obj.match(self.desc_2)
        matches = sorted(matches, key=lambda x: x.distance)
        percentage_similarity = 0

        for i in range(0,len(matches)):
            p1 = matches[i].distance
            if p1 <= 250:
                percentage_similarity += 1
        percentage_similarity = float(percentage_similarity / len(matches) * 100)

        return percentage_similarity


    def get_image_cluster(self,num_clusters,center_index):
        max_per = 0

        for n in range(num_clusters):
                print("\n --- Images from cluster #%d ---" % n)
                name = "{}_{}".format("cluster_num", n)
                name = []
                cluster_center = self.cluster[center_index[n]]
                print("Image at center %s" % images[center_index[n]])
                for i in np.argwhere(cluster == n):
                    j = i[0]
                    name.append(j)
            
                image_desc = desc_dict[images[center_index[n]]]
                print("for image::::::::",images[center_index[n]])
                print("--------------d$$$$$$$$$$$$$$$$$$$---------------")
                percentage_similarity = self.get_percent_similarity(image_desc)
                print(percentage_similarity)

                if percentage_similarity == 100:
                    print("Image is the one %s" % images[center_index[n]])
                    exit()
                
                if max_per < percentage_similarity:
                    max_per =  percentage_similarity
                    array_max = name
                    cluster_num = n
        return array_max, cluster_num
    
    def execute(self,num_clusters):
        array_max, cluster_num = self.get_image_cluster(num_clusters,self.center_index)
        max_per = 0 

        # print(name)
        print("-------------")
        print(array_max)
        print("Image matching for cluster %d",cluster_num)
        for p in range(len(array_max)):

            print("Image %s" % images[array_max[p]])
            image_desc = desc_dict[images[array_max[p]]]
            percentage_similarity = get_percent_similarity(image_desc)
            if max_per < percentage_similarity:
                max_per = max(max_per, percentage_similarity)
                real_image = images[array_max[p]]
                
        print("we think the match is ::::" + real_image)
        print("le temps maximum est ", (time.time() - start_time))

        with open('save_test.csv', 'a') as csvfile:
            fieldnames = ["im_typ", "sim_image"]
            writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow({"im_typ": test_image,"sim_image": real_image})


hell = Matching_Algorithm('sift','../Work/original_images/adam_orig.jpeg','bf')
hell.execute(num_clusters)
# print(hell)