import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('home1.jpg', 0)  # queryImage

# Initiate surf detector
surf = cv2.xfeatures2d.SURF_create()

# find the keypoints and descriptors with surf
kp1, des1 = surf.detectAndCompute(img1, None)