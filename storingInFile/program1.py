import numpy as np
import cv2
import tables
from matplotlib import pyplot as plt

img1 = cv2.imread('../images/home/home1.jpg', 0)  # queryImage

# Initiate surf detector
surf = cv2.ORB_create()

# find the keypoints and descriptors with surf
kp1, des1 = surf.detectAndCompute(img1, None)


#data = np.linspace(0,1,2)

#np.savetxt('A_data.dat', des1 )



#my_arrays = [np.ones((2, 2, 4098)) for x in range(10)]
my_arrays = des1

h5_file = tables.open_file('my_array.h5', mode='w', titel='many large arrays')

for n, arr in enumerate(my_arrays):
    h5_file.create_array('/', 'my_array{}'.format(n), arr)
h5_file.close()

#########


