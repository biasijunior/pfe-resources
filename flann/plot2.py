import matplotlib.pyplot as plt
import numpy as np
# AKAZE KNN SIMILARITY

# plt.plot(['original','cropped','lumino','rot_90','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,90,	60, 56, 38,	5,	0,	0,	0,	0,	0,	0,	0,	0,	0])
# plt.plot(['original','cropped','lumino','rot_90','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,91,	66, 62, 44,	6,	0,	0,	0,	0,	0,	0,	0,	0,	0])
# plt.plot(['original','cropped','lumino','rot_90','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,92,	71, 67, 50,	7,	0,	0,	0,	0,	0,	0,	0,	0,	0])
# plt.plot(['original','cropped','lumino','rot_90','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,93,	76, 72, 56,	8,	0,	0,	0,	0,	0,	0,	0,	0,	0])
# plt.plot(['original','cropped','lumino','rot_90','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,93, 79, 76, 60, 10,	0,	0,	0,	0,	0,	0,	0,	0,	0])
# plt.plot(['original','cropped','lumino','rot_90','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,94, 83, 80, 65, 11,	0,	0,	0,	0,	1,	0,	0,	0,	0])
# plt.plot(['original','cropped','lumino','rot_90','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,95, 86, 83, 70, 13,	1,	1,	1,	1,	2,	1,	1,	1,	1])
# plt.plot(['original','cropped','lumino','rot_90','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,95, 88, 86, 74, 18,	3,	4,	4,	4,	7,	3,	5,	4,	5])
# plt.plot(['original','cropped','lumino','rot_90','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,96, 91, 89, 81, 28,	12,	12,	13,	12,	16,	11,	14,	12,	15])
# plt.plot(['original','cropped','lumino','rot_90','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,97, 95, 93, 88, 50,	37,	36,	37,	34,	40,	36,	39,	35,	38])
# plt.plot(['original','cropped','lumino','rot_90','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100])

# plt.legend(['0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95','1'], loc='lower left')
# plt.axis([None,None, 0, 101])
# plt.ylabel('percentage similarity')
# plt.xlabel(' images ')
# # plt.title('akaze_bf_knn_similarity_percentage')
# plt.show()

###############################

#orb bf knn similarity

<<<<<<< HEAD
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','2','3','4','5','6','7','8','9'],[100,	100,	54,	22,	5,	0,	0,	0,	0,	0,	0,	0,	0,	9])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','2','3','4','5','6','7','8','9'],[100,	100,	57,	25,	9,	0,	0,	0,	0,	0,	0,	0,	0,	13])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','2','3','4','5','6','7','8','9'],[100,	100,	60,	30,	12,	0,	0,	0,	0,	0,	0,	0,	0,	16])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','2','3','4','5','6','7','8','9'],[100,	100,	62,	34,	19,	0	,0,	0,	0,	0,	0,	0,	0,	23])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','2','3','4','5','6','7','8','9'],[100,	100,	65,	37,	23,	0	,0,	1,	0,	0,	0,	0,	0,	31])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','2','3','4','5','6','7','8','9'],[100,	100,	67,	43,	30,	2	,2,	1,	2,	0,	1,	0,	1,	38])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','2','3','4','5','6','7','8','9'],[100,	100,	70,	47,	38,	5	,4,	3,	5,	0,	4,	0,	2,	46])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','2','3','4','5','6','7','8','9'],[100,	100,	74,	53,	46,	9	,10,	7	,10	,0,	11,	0,	6,	56])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','2','3','4','5','6','7','8','9'],[100,	100,	79,	63,	59,	23,	23,	18,	21,	5	,23	,0	,17,	66])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','2','3','4','5','6','7','8','9'],[100,	100,	86,	79,	76,	48,	51,	46,	46,	29,48	,6	,42,	80])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','2','3','4','5','6','7','8','9'],[100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100])
plt.plot(['condame_90.jpg', 'condame.jpg', 'condame_cropped.jpg', 'condame_lumino.jpg', 'condame_45.png', '1', '2', '3',      '4', '5', '6', '7', '8', '9'],[99, 99, 72, 55, 47, 2, 2, 1, 1, 0, 0, 0, 0, 0])
plt.plot(['condame_90.jpg', 'condame.jpg', 'condame_cropped.jpg', 'condame_lumino.jpg', 'condame_45.png', '1', '2', '3',      '4', '5', '6', '7', '8', '9'],[99, 99, 75, 58, 52, 3, 2, 2, 2, 1, 1, 0, 0, 0])
plt.plot(['condame_90.jpg', 'condame.jpg', 'condame_cropped.jpg', 'condame_lumino.jpg', 'condame_45.png', '1', '2', '3','4', '5', '6', '7', '8', '9'],[99, 99, 76, 61, 60, 6, 6, 4, 3, 3, 2, 0, 0, 0])
plt.plot(['condame_90.jpg', 'condame.jpg', 'condame_cropped.jpg', 'condame_45.png', 'condame_lumino.jpg', '1', '2', '3', '4','5', '6', '7','8', '9'],[100, 100, 79, 66, 65, 11, 10, 8, 7, 7, 6, 2, 0, 0])
plt.plot(['condame_90.jpg', 'condame.jpg', 'condame_cropped.jpg', 'condame_45.png', 'condame_lumino.jpg', '1','2', '3', '4', '5', '6', '7','8', '9'],[100, 100, 82, 70, 69, 20, 19, 16, 15, 15, 14, 1, 0, 0])
plt.plot(['condame.jpg', 'condame_90.jpg', 'condame_cropped.jpg', 'condame_lumino.jpg', 'condame_45.png', '1','2', '3', '4',
        '5', '6', '7', '8', '9'],[100, 99, 85, 77, 77, 36, 31, 30, 28, 28, 26, 2, 1, 0])
plt.plot(['condame_90.jpg', 'condame.jpg', 'condame_cropped.jpg', 'condame_45.png', 'condame_lumino.jpg', '1', '2', '3', '5',
     '5', '6', '7', '8', '9'],[100, 100, 91, 88, 86, 59, 59, 55, 55, 53, 52, 3, 3, 2])

plt.legend(['0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'], loc='lower left')
plt.axis([None,None, 0, 101])
plt.ylabel('percentage similarity')
plt.xlabel(' images ')
# plt.title('akaze_bf_knn_similarity_percentage')
plt.show()
plt.savefig('../database/bf_results/knn_matching_time_algos.png')
=======
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	54,	    22, 	5,	    0,	    0,	    0,	    0,	    0,	    0,  	0,	    0,	9])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	57,	    25, 	9,	    0,	    0,	    0,	    0,	    0,	    0,  	0,	    0,	13])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	60,	    30, 	12, 	0,	    0,	    0,	    0,	    0,	    0,  	0,	    0,	16])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	62,	    34, 	19, 	0,      0,	    0,	    0,	    0,	    0,  	0,	    0,	23])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	65,	    37, 	23, 	0,      0,	    1,	    0,	    0,	    0,  	0,	    0,	31])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	67,	    43, 	30, 	2,      2,	    1,	    2,	    0,	    1,  	0,	    1,	38])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	70,	    47, 	38, 	5,      4,	    3,	    5,	    0,	    4,  	0,	    2,	46])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	74,	    53, 	46, 	9,      10,	    7,      10,     0,	    11, 	0,	    6,	56])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	79,	    63, 	59, 	23,	    23,	    18,	    21,	    5,      23,     0,      17,	66])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	86,	    79, 	76, 	48,	    51,	    46,	    46,	    29,     48,     6,      42,	80])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	100,    100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100])

# plt.legend(['0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95','1'], loc='lower left')
# plt.axis([None,None, 0, 101])
# plt.ylabel('percentage similarity')
# plt.xlabel(' images ')
# # plt.title('akaze_bf_knn_similarity_percentage')
# plt.show()
# # plt.savefig('../database/bf_results/knn_matching_orb.png')

######################################

# sift bf knn similarity

# plt.plot(['original','cropped','rot_90','lumino','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	88,	82,	67,	56,	10,	0,	0,	0,	0,	0,	0,	0,	0,	0])
# plt.plot(['original','cropped','rot_90','lumino','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	88,	83,	69,	59,	12,	1,	0,	0,	1,	0,	0,	0,	0,	0])
# plt.plot(['original','cropped','rot_90','lumino','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	89,	85,	70,	63,	15,	2,	0,	0,	2,	0,	0,	0,	0,	0])
# plt.plot(['original','cropped','rot_90','lumino','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	89,	86,	71,	65,	18,	2,	0,	0,	2,	1,	0,	0,	0,	0])
# plt.plot(['original','cropped','rot_90','lumino','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	89,	86,	74,	68,	20,	4,	0,	1,	3,	1,	0,	1,	1,	0])
# plt.plot(['original','cropped','rot_90','lumino','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	90,	88,	76,	70,	23,	6,	2,	2,	4,	3,	2,	2,	3,	2])
# plt.plot(['original','cropped','rot_90','lumino','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	90,	89,	78,	73,	29,	10,	4,	5,	6,	6,	5,	4,	5,	4])
# plt.plot(['original','cropped','rot_90','lumino','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	92,	90,	80,	77,	36,	15,	8,	11,	10,	13,	12,	10,	11,	9])
# plt.plot(['original','cropped','rot_90','lumino','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	93,	92,	84,	81,	45,	26,	20,	23,	20,	26,	26,	22,	22,	21])
# plt.plot(['original','cropped','rot_90','lumino','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	95,	94,	89,	89,	61,	49,	46,	44,	43,	48,	49,	48,	49,	49])
# plt.plot(['original','cropped','rot_90','lumino','rot_45','blurred','other1','other2','other3','other4','other5','other6','other7','other8','other9'],[100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100])

# plt.legend(['0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95','0.9','1'], loc='lower left')
# plt.axis([None,None, 0, 101])
# plt.ylabel('percentage similarity')
# plt.xlabel(' images ')
# # plt.title('akaze_bf_knn_similarity_percentage')
# plt.show()


############################

# surf bf knn similarity

# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8'],[100, 100,	100,	100,	20,	0,	0,	0,	0,	0,	20,	0,	0])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8'],[100, 100,	100,	100,	20,	0,	0,	0,	0,	0,	20,	0,	0])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8'],[100, 100,	100,	100,	20,	0,	0,	0,	0,	0,	20,	0,	0])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8'],[100, 100,	100,	100,	40,	0,	0,	0,	0,	0,	20,	0,	0])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8'],[100, 100,	100,	100,	40,	0,	0,	0,	0,	0,	20,	0,	0])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8'],[100, 100,	100,	100,	40,	0,	0,	0,	0,	0,	20,	20,	0])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8'],[100, 100,	100,	100,	40,	0,	0,	0,	20,	0,	20,	20,	0])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8'],[100, 100,	100,	100,	40,	0,	0,	0,	60,	40,	20,	20,	0])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8'],[100, 100,	100,	100,	80,	0,	0,	60,	80,	40,	40,	20,	0])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8'],[100, 100,	100,	100,	80,	20,	20,	80,	80,	40,	60,	20,	20])
# plt.plot(['original','rot_90','cropped','lumino','rot_45','other1','other2','other3','other4','other5','other6','other7','other8'],[100, 100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100,	100])

# plt.legend(['0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95','1'], loc='lower left')
# plt.axis([None,None, 0, 101])
# plt.ylabel('percentage similarity')
# plt.xlabel(' images ')
# # plt.title('akaze_bf_knn_similarity_percentage')
# plt.show()

>>>>>>> 2b6486dfb15b4d8e13b2c293cfc796f4a0132278
##################################################

# bf KNN COMPUTE TIME for matching

# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],[0.460099991,	0.579933325,	0.506799976,	0.502633333,	0.5454,	0.541166663,	0.52743334,	0.516600021,	0.496799986,	0.522266674])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],[0.015199995,	0.0033,     	0.0161003,      0.0150995,  	0.00920002, 	0.00880001, 	0.00879998, 	0.01749997, 	0.00860004, 	0.00630002])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],[0.28216664,	0.213866663,	0.212300022,	0.223466682,	0.214866678,	0.227499986,	0.226100039,	0.213566669,	0.220033328,	0.215566683])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],[0.071000099,	0.000249982,	0.000500083,	0.00000012, 	0.000499964,	0.018000007,	0.000249982,	0.004000008,	0.004499972,	0.00150001])

# plt.legend(['AKAZE','ORB','SIFT','SURF'], loc='lower left')
# plt.axis([0.2, 1, 0, 1])
# plt.ylabel('computational time for matching (sec)')
# plt.xlabel(' distance ')
# plt.show()

# ################################################


# time for matching algos comparing using BFMATCHER_KNN

# plt.plot(['original','rotated_45','rotated_90','cropped','blurred','luminous'],[0.80275005102125, 1.46306690176328,  1.32577413121859,    1.308867122729620,   0.246950163443883, 0.730999946594])
# plt.plot(['original','rotated_45','rotated_90','cropped','blurred','luminous'],[0.0153332551320667, 0.00503451390699907, 0.00524500716816295, 0.00508482672951438, 0.000804948806762695, 0.00999999046326])
# plt.plot(['original','rotated_45','rotated_90','cropped','blurred','luminous'],[0.713333288827, 0.83459939,  0.748774860302607,   0.744147501389186,  0.110249926646550, 1.07600021362])
# plt.plot(['original','rotated_45','rotated_90','cropped','blurred','luminous'],[0.0166654586791667, 0.000416656335195167,    0.0010000268618275,    0.000999987125396667,    0.000500003496805, 0.00400018692017])

# plt.legend(['AKAZE','ORB','SIFT','SURF'], loc='lower left')
# plt.axis([None,None, 0, 2])
# plt.ylabel('computational_time_average (sec)')
# plt.xlabel(' modified images ')
# # plt.title('algos_KNN_matching_time')
# plt.show()
# plt.pause(0.05)
#plt.savefig('../database/bf_results/knn_matching_time_algos.png')

# ###############################################
# time for getting descriptors knn comparing algos

# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[7.16617709000905, 2.94264774322510, 2.94371773600578, 2.83267605900764])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[0.431859237497503, 0.203161426024003, 0.206230330467224, 0.187110176086426])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[14.33533716, 3.38424399693807, 3.36949411431948, 3.05343625942866])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[6.25990654627482, 4.66048846046130, 4.56513202985128, 2.80168905258179])

# plt.legend(['AKAZE','ORB','SIFT','SURF'], loc='lower left')
# plt.axis([None,None, 0, 15])
# plt.ylabel('getting_descriptors_time_average (sec)')
# plt.xlabel(' modified images ')
# plt.title('algos_BF_KNN_desc')
# plt.show()
# # plt.pause(0.05)
# # plt.savefig('../database/bf_results/knn_descriptors_time_algos.png')


# ###########################################

# similarity percentage comparing between algos _KNN

plt.plot(['original','rotated_45','rotated_90','cropped','blurred','luminous'],[100, 55.8822719812, 71.1641894623, 86.1820443954, 41.2494103330, 71])
plt.plot(['original','rotated_45','rotated_90','cropped','blurred','luminous'],[100, 19.1, 91.0, 65, 97, 30])
plt.plot(['original','rotated_45','rotated_90','cropped','blurred','luminous'],[100, 57.80013289, 80.5709513950, 81.8748116254, 52.8376568889, 70])
plt.plot(['original','rotated_45','rotated_90','cropped','blurred','luminous'],[100, 39.33333333333333, 85.66666666666667, 81, 12.75, 100])

plt.legend(['AKAZE','ORB','SIFT','SURF'], loc='lower left')
plt.axis([None,None, 0, 101])
plt.ylabel('percentage similarity')
plt.xlabel(' modified images ')
# plt.title('knn_similarity_percentage')
plt.show()
# plt.pause(0.05)
# plt.savefig('../database/bf_results/knn_matching_time_algos.png')


# #############################################

# time for matching algos comparing using FLANN

# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[1.817637318, 1.70381494, 1.671470871, 0.256204659])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[24.4774505, 28.31613097, 28.51775557, 1.759947517 ])
# # plt.plot(['rotated_45','rotated_90','cropped','blurred'],[0.83459939,  0.748774860302607,   0.744147501389186,  0.110249926646550])
# # plt.plot(['rotated_45','rotated_90','cropped','blurred'],[10.8116898179054,    12.5201132694880,    12.4711701969306,    0.772699435551961])

# plt.legend(['SIFT','SURF'], loc='lower left')
# plt.axis([None,None, 0, 30])
# plt.ylabel('computational_time_average (sec)')
# plt.xlabel(' modified images ')
# plt.title('algos_Flann_matching_time')
# plt.show()
# # plt.pause(0.05)
# #plt.savefig('../database/bf_results/knn_matching_time_algos.png')

# ########################################

# time for getting descriptors FLANN comparing algos

# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[13.18976329, 2.969968061, 2.905574723, 2.722227371])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[5.70571029, 4.193383998, 4.149927938, 2.519072461])
# # plt.plot(['rotated_45','rotated_90','cropped','blurred'],[14.33533716, 3.38424399693807, 3.36949411431948, 3.05343625942866])
# # plt.plot(['rotated_45','rotated_90','cropped','blurred'],[6.25990654627482, 4.66048846046130, 4.56513202985128, 2.80168905258179])

# plt.legend(['SIFT','SURF'], loc='lower left')
# plt.axis([None,None, 0, 15])
# plt.ylabel('getting_descriptors_time_average (sec)')
# plt.xlabel(' modified images ')
# plt.title('algos_BF_FLANN_desc')
# plt.show()
# # plt.pause(0.05)
# # plt.savefig('../database/bf_results/knn_descriptors_time_algos.png')

# #############################################

# similarity percentage comparing between algos _FLANN

# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[57.80013289, 80.57095139, 81.87481163, 52.83765689])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[15.25934353, 91.94862396, 74.87631276, 34.13645978])
# # plt.plot(['rotated_45','rotated_90','cropped','blurred'],[57.80013289, 80.5709513950, 81.8748116254, 52.8376568889])
# # plt.plot(['rotated_45','rotated_90','cropped','blurred'],[15.25934353435, 91.94862396455, 74.87631275951, 34.1364597792])

# plt.legend(['SIFT','SURF'], loc='lower left')
# plt.axis([None,None, 0, 100])
# plt.ylabel('percentage similarity')
# plt.xlabel(' modified images ')
# plt.title('flann_similarity_percentage')
# plt.show()
# # plt.pause(0.05)
# # plt.savefig('../database/bf_results/knn_matching_time_algos.png')

# #######################################

# FLANN KNN ORB

# plt.plot(['condame_original','condame_90','condame_cropped','condame_45','prayer_45','life','rose_90','malcomx','prayer_blurred','rose_blurred'],[100,100,68,25,0,0,0,0,0,0])
# plt.plot(['condame_original','condame_90','condame_cropped','condame_45','prayer_45','life','rose_90','malcomx','prayer_blurred','rose_blurred'],[100,100,69,35,0,0,0,0,0,0])
# plt.plot(['condame_original','condame_90','condame_cropped','condame_45','prayer_45','life','rose_90','malcomx','prayer_blurred','rose_blurred'],[100,100,69,42,0,0,0,0,0,0])
# plt.plot(['condame_original','condame_90','condame_cropped','condame_45','prayer_45','life','rose_90','malcomx','prayer_blurred','rose_blurred'],[100,100,71,48,1,0,0,0,0,0])
# plt.plot(['condame_original','condame_90','condame_cropped','condame_45','prayer_45','life','rose_90','malcomx','prayer_blurred','rose_blurred'],[100,100,74,54,2,0,1,2,0,0])
# plt.plot(['condame_original','condame_90','condame_cropped','condame_45','prayer_45','life','rose_90','malcomx','prayer_blurred','rose_blurred'],[100,100,77,59,5,2,4,2,0,0])
# plt.plot(['condame_original','condame_90','condame_cropped','condame_45','prayer_45','life','rose_90','malcomx','prayer_blurred','rose_blurred'],[100,100,79,67,8,4,10,5,0,0])
# plt.plot(['condame_original','condame_90','condame_cropped','condame_45','prayer_45','life','rose_90','malcomx','prayer_blurred','rose_blurred'],[100,100,82,72,17,12,17,12,0,0])
# plt.plot(['condame_original','condame_90','condame_cropped','condame_45','prayer_45','life','rose_90','malcomx','prayer_blurred','rose_blurred'],[100,100,85,80,33,27,32,25,1,0])
# plt.plot(['condame_original','condame_90','condame_cropped','condame_45','prayer_45','life','rose_90','malcomx','prayer_blurred','rose_blurred'],[100,100,91,88,58,52,59,53,1,1])
# plt.plot(['condame_original','condame_90','condame_cropped','condame_45','prayer_45','life','rose_90','malcomx','prayer_blurred','rose_blurred'],[100,100,98,97,92,92,92,92,2,5])

# plt.legend(['0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95','1'], loc='lower left')
# plt.axis([None,None, 0, 100])
# plt.ylabel('percentage similarity')
# plt.xlabel(' images ')
# plt.title('flann_orb_similarity_percentage')
# plt.show()
# # plt.pause(0.05)
# # plt.savefig('../database/bf_results/knn_matching_time_algos.png')
