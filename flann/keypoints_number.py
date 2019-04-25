import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time
import winsound
import functions as fn




akaze = cv2.xfeatures2d.SURF_create()
# image =[]

all_images = fn.loadimages("../images/testBooks/test/*")

for images, title in all_images:
    kp , desc= akaze.detectAndCompute(images, None)
    bf = cv2.BFMatcher()
    number_key_points=len(kp)
    number_descriptors=len(desc)
    print('%s'%title)
    print('kp: %d'%number_key_points )
    print('desc: %d'%number_descriptors+'\n')