import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time
import winsound
import functions as fn


kaze = cv2.xfeatures2d.SURF_create(10000)
# kaze = cv2.AKAZE_create()
all_images = fn.loadimages("../images/testBooks/test/*")

for image_to_compare, title in all_images:
    # total_time=time.time()
    kp_1, desc_1 = kaze.detectAndCompute(image_to_compare, None)
    print("title: %s"%title)
    print("nombre %s"%len(kp_1))
    # print("somme : %s"%(time.time()-total_time))  
  
  
winsound.MessageBeep()
