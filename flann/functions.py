import glob
import cv2
import sys
import csv



def loadimages(path_to_images):
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


def save_stats_to_file(file_name,zipped_file):

    im_typ = 'image type'
    percent_sim = 'percentage similarity'
    compute_time = 'computational time'
    with open('../database/' + file_name, 'a') as csvfile:
        fieldnames = [im_typ, percent_sim, compute_time]

        writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
        writer.writeheader()
        for img_type, per_sim, time_taken in zipped_file:
            writer.writerow(
                {im_typ: img_type , percent_sim : per_sim, compute_time: time_taken})
    print('finished writing to a file')
    print('Done!!!')

# sift_results.csv
