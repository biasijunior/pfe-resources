import cv2
import glob
import sys
import csv
import time

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
            d = 0.6
        else:
            d = provided_distance
        percent = []
        image = []
        compute_time_arry = []
        all_images_to_compare = self.loadimages()
        kp_1, desc_1 = self.get_keypoint_and_desc()
        print('comparing image descriptors for distance ' + str(d))

        for image_to_compare, title in all_images_to_compare:
            kp_2 , desc_2 = self.algorithm.detectAndCompute(image_to_compare, None)
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
            percent.append(str(int(percentage_similarity)))
            image.append(title)
            compute_time_arry.append(total_time)

        self.zipped = zip(image, percent, compute_time_arry)
        return self.zipped
    
    def save_stats_to_file(self,file_name):
        zipped_file = self.compare_images()
        im_typ = 'image type'
        percent_sim = 'percentage similarity'
        compute_time = 'computational time'
        file_name = '../database/' + file_name + "_stats_"+self.matcher_name + "_matcher.csv"

        with open(file_name, 'a') as csvfile:
            fieldnames = [im_typ, percent_sim, compute_time]

            writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
            if csvfile.tell() ==0:
                writer.writeheader()
            for img_type, per_sim, time_taken in zipped_file:
                writer.writerow({im_typ: img_type, percent_sim: per_sim, compute_time: time_taken})
        print('finished saving  to file')
        print('Done!!!')


# m = Matching_Algorithm('orb', "../images/train/arabic.jpg","bf", "../images/train/arabic.jpg")
# @TODO ikram look at the syntax the way to initialise it

# Matching_Algorithm("algorithm_to_use", "image_url", "bf_or_flann_matcher", "get_train_images_url")
# m.save_stats_to_file("biasi")
# p = Matching_Algorithm.
algo = ['sift', 'surf', 'orb', 'akaze']

for algo_name in algo:
     sift = Matching_Algorithm(algo_name, "../images/train/arabic.jpg", "bf", "../images/train/arabic.jpg")
     print algo_name
     for i in range(0, 2):
        sift.save_stats_to_file(algo_name)

# sift.loadimages("../images/test/original_book.jpg")
