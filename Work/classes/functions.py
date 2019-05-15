import glob
import cv2
import sys
import csv
import time


def loadimages(path_to_images):
    all_images_to_compare = []
    titles = []
    i = 1

    print("loading images \r")

    for f in glob.iglob(path_to_images):
        # print f
        image = cv2.imread(f,0)
        # image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        titles.append(f.rsplit('/', 1)[1])
        print (i),
        sys.stdout.flush()
        all_images_to_compare.append(image)
        i += 1
    
    return zip(all_images_to_compare, titles)


def save_stats_to_file(file_name,zipped_file):

    im_typ = 'image name'
    # comment/uncomment where theres percentage similarity
    # percent_sim = 'percentage similarity'
    compute_time = 'matching time'
    desc_time = 'time_to_fetch_descriptors'


    with open('../database/' + file_name, 'a') as csvfile:
        # fieldnames = [im_typ, percent_sim, compute_time,desc_time]
        fieldnames = [im_typ, compute_time, desc_time]


        writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
        if csvfile.tell == 0:
            writer.writeheader()
        
        for img_type,  match_time, desc_time_taken in zipped_file:

        # for img_type, per_sim, match_time, desc_time_taken in zipped_file:
            writer.writerow(
                # {im_typ: img_type, percent_sim: per_sim, compute_time: match_time, desc_time: desc_time_taken})

                {im_typ: img_type , compute_time: match_time, desc_time: desc_time_taken})
    print('finished writing to a file')
    print('Done!!!')


def run(obj):

    sift = cv2.ORB_create()
    bf = cv2.BFMatcher()
    
    print('comparing...')
    percent = []
    image = []
    compute_time_arry = []
    all_images_to_compare = loadimages("../images/train/*")

    for image_to_compare, title in all_images_to_compare:
        start_time = time.time()
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

        matches = bf.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6*n.distance:
                good_points.append(m)
        number_keypoints = 0
        number_keypoints = 0
        if len(desc_2) <= len(desc_1):
            number_keypoints = len(desc_2)
        else:
            number_keypoints = len(des_1)

        print("Title: " + title)
        percentage_similarity = float(len(good_points)) / number_keypoints * 100
        total_time = time.time() - start_time
        print("--- %s seconds ---" % (total_time))
        print("Similarity: " + str((percentage_similarity)) + " %\n")
        percent.append(str(int(percentage_similarity)))
        image.append(title)
        compute_time_arry.append(total_time)

        # pprint.pprint(data1)
    # zipped = sorted(zip(percent, image), key=lambda pair: pair[0], reverse= True)
    # zipped = sorted(zipped, key = lambda x: x[0])
    zipped = zip(image, percent, compute_time_arry)

    # sift_results.csv
