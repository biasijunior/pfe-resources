import os
import cv2
import glob
import sys
import csv
import time
import os

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
            else:
                raise Exception("object not created")
            
            print algorithm + " object successfully created"
        except Exception as error:
            print repr(error) + "\nError on algorith_name, ENTER a valid algorithm name e.g ORB, SIFT etc"
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
            print "You have chosen "+ matcher_obj + "matcher"
        except Exception as error:
            print repr(error) + "\n enter bf or flann"
            exit()



    def get_keypoint_and_desc(self):
       self.kp_1 , self.desc_1 = self.algorithm.detectAndCompute(self.image, None)
       return self.kp_1, self.desc_1

    def loadimages(self,path_to_images=""):
        if path_to_images == "":
            if self.train_images_url != "":
               path_to_images = self.train_images_url
            else:
                path_to_images = raw_input("Please provide the url to your train images:")
        all_images_to_compare = []
        titles = []
        i = 1
        print("loading images \r")

        for f in glob.iglob(path_to_images):
            imag = cv2.imread(f)
            image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
            titles.append(f.rsplit('/', 1)[1])
            print i,
            sys.stdout.flush()
            all_images_to_compare.append(image)
            i += 1

        return zip(all_images_to_compare, titles)

    def compare_images(self,provided_distance=""):
        if provided_distance =="":
            d = 0.65
        else:
            d = provided_distance
        percent = []
        image = []
        compute_time_arry = []
        get_desc_time = []
        all_images_to_compare = self.loadimages()
        kp_1, desc_1 = self.get_keypoint_and_desc()
        print('comparing image descriptors for distance ' + str(d))

        for image_to_compare, title in all_images_to_compare:
            desc_extract_time = time.time()
            kp_2 , desc_2 = self.algorithm.detectAndCompute(image_to_compare, None)
            get_desc_time.append(time.time() - desc_extract_time)

            start_time = time.time()
            matches = self.matcher_obj.knnMatch(self.desc_1, desc_2, k=2)

            good_points = []
            for m, n in matches:
                if m.distance < d*n.distance:
                    good_points.append(m)
            number_keypoints = 0
            if len(desc_2) <= len(self.desc_1):
                number_keypoints = len(desc_2)
            else:
                number_keypoints = len(self.desc_1)

            print("Title: " + title)
            percentage_similarity = float(len(good_points)) / number_keypoints * 100
            total_time = time.time() - start_time
            print("--- %s seconds ---" % (total_time))
            print("Similarity: " + str((percentage_similarity)) + " %\n")
            percent.append(str(percentage_similarity))
            image.append(title)
            compute_time_arry.append(total_time)

        self.zipped = zip(image, compute_time_arry, get_desc_time, percent)
        return self.zipped
    
    def save_stats_to_file(self,file_name):
        zipped_file = self.compare_images()
        im_typ = 'image type'
        percent_sim = 'percentage similarity(%)'
        compute_time = 'computational time'
        desc_time = 'Time to fetch descriptors'
        file_name = '../database/' + file_name + "_stats_"+self.matcher_name + "_matcher.csv"

        with open(file_name, 'a') as csvfile:
            fieldnames = [im_typ, compute_time, desc_time, percent_sim]

            writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
            #to check if the header is already returned
            if csvfile.tell() ==0:
                writer.writeheader()
            for img_type, time_taken, descrp_time, per_sim in zipped_file:
                writer.writerow(
                    {im_typ: img_type, compute_time: time_taken, desc_time: descrp_time ,percent_sim: per_sim})
        print('finished saving  to file')
        print('Done!!!')


# m = Matching_Algorithm('orb', "../images/train/arabic.jpg","bf", "../images/train/arabic.jpg")
# @TODO ikram look at the syntax the way to initialise it

# Matching_Algorithm("algorithm_to_use", "image_url", "bf_or_flann_matcher", "get_train_images_url")
# m.save_stats_to_file("biasi")
# p = Matching_Algorithm.
get_time = time.time()
algo = ['sift', 'surf', 'akaze', 'orb']
# algo = ['orb', 'akaze']
# algo = ['akaze']

for algo_name in algo:
     
    # sift = Matching_Algorithm(algo_name, "../images/train/arabic.jpg", "bf", "../images/testBooks/arabic/*")
    # sift = Matching_Algorithm(algo_name, "../images/train/butterfly.jpg", "bf", "../images/testBooks/butterfly/*")
    # sift = Matching_Algorithm(algo_name , "../images/train/condame.jpg", "bf", "../images/testBooks/condame/*")
    # sift = Matching_Algorithm(algo_name, "../images/train/life.JPEG", "bf", "../images/testBooks/lives/*")
    # sift = Matching_Algorithm(algo_name, "../images/train/likeblack.jpg", "bf", "../images/testBooks/likeblack/*")
    # sift = Matching_Algorithm(algo_name, "../images/train/madam.jpg", "bf", "../images/testBooks/madam/*")
    # sift = Matching_Algorithm(algo_name, "../images/train/malcomx.jpg", "bf", "../images/testBooks/malcomx/*")
    # sift = Matching_Algorithm(algo_name, "../images/train/memory.jpg", "bf", "../images/testBooks/memory/*")
    # sift = Matching_Algorithm(algo_name, "../images/train/prayer.jpg", "bf", "../images/testBooks/prayer/*")
    # sift = Matching_Algorithm(algo_name, "../images/train/rose.jpg", "bf", "../images/testBooks/rose/*")
    # sift = Matching_Algorithm(algo_name, "../images/train/sherlock.jpg", "bf", "../images/testBooks/sherlock/*")
    sift = Matching_Algorithm(algo_name, "../images/train/the_100.jpg", "bf", "../images/testBooks/100/*")
    print algo_name
    for i in range(0, 10):
        print("--- %s iteration ---" % (i + 1))
        sift.save_stats_to_file(algo_name)

os.system('afplay /System/Library/Sounds/Sosumi.aiff')
get_time = time.time() - get_time 
# print "time taken is " + str(get_time)
print("time taken is:  %s seconds!" % (get_time))
# sift.loadimages("../images/test/original_book.jpg")
