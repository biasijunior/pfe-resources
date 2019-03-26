import numpy as np
import cv2
from matplotlib import pyplot as plt
#read image
img1 = cv2.imread('../images/books/book1.jpg', 0)
#detectors
orb = cv2.ORB_create()
#descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
#array
#data = np.linspace(10,20,5)

"""x = np.linspace(10, 20, 5)
y = np.random.random(5)

data = np.column_stack((x, y))
#storing in a file
header ="X-Column, Y-Column"
header+="\n this is"
#np.savetxt('A_data.dat', data, header=header)
f = open('AD_data.dat', 'wb')
np.savetxt(f, [], header=header)
for i in range(5):
    data = np.column_stack((x[i], y[i]))
    np.savetxt(f, data)
f.close()"""





with open('BB_data.dat', 'w') as f:
    f.write('# This is the header')
    for data in kp1,des1:
        f.write(str(data))