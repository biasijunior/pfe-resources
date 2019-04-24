import matplotlib.pyplot as plt
import numpy as np
#AKAZE KNN SIMILARITY
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[100,100,100,100,100,100,100,100,100,100,100])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[38,44,50,56,60,65,70,74,81,88,100])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[56,	62,	67,	72,	76,	80,	83,	86,	89,	93,	100])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[90,	91,	92,	93,	93,	94,	95,	95,	96,	97,	100])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[0,	0,	0,	0,	0,	0,	1,	4,	12,	36,	100])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[0,	0,	0,	0,	0,	0,	1,	4,	13,	37,	100])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[0,	0,	0,	0,	0,	0,	1,	4,	12,	34,	100])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[0,	0,	0,	0,	0,	1,	2,	7,	16,	40,	100])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[0,	0,	0,	0,	0,	0,	1,	3,	11,	36,	100])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[0,	0,	0,	0,	0,	0,	1,	5,	14,	39,	100])

# plt.legend(['img_original', 'rotated_45', 'rotated_90', 'cropped','other1', 'other2','other3','other4','other5','other6'], loc='lower left')

# plt.axis([0.2, 1, 0, 120])
# plt.ylabel('similarity_percentage')
# plt.xlabel(' distance')
# plt.show()

##################################################

# AKAZE KNN COMPUTE TIME

# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],[0.460099991,	0.579933325,	0.506799976,	0.502633333,	0.5454,	0.541166663,	0.52743334,	0.516600021,	0.496799986,	0.522266674])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],[0.009266671,	0.005666677,	0.006466667,	0.006966694,	0.007699998,	0.005233343,	0.004166659,	0.008500004,	0.006833339,	0.006833323])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],[0.28216664,	0.213866663,	0.212300022,	0.223466682,	0.214866678,	0.227499986,	0.226100039,	0.213566669,	0.220033328,	0.215566683])
# plt.plot([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],[5.468866666,	5.432550001,	5.440433296,	5.28344996,     5.693766673,	5.343166677,	5.267600012,	5.119233354,	4.686399992,	4.956800032])

# plt.legend(['AKAZE','ORB','SIFT','SURF'], loc='lower left')
# plt.axis([0.2, 1, 0, 7])
# plt.ylabel('computational_time_average (sec)')
# plt.xlabel(' distance ')
# plt.show()

#################################################


#time for matching algos comparing using BFMATCHER_KNN

# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[1.46306690176328,  1.32577413121859,    1.308867122729620,   0.246950163443883])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[0.00503451390699907, 0.00524500716816295, 0.00508482672951438, 0.000804948806762695])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[0.83459939,  0.748774860302607,   0.744147501389186,  0.110249926646550])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[10.8116898179054,    12.5201132694880,    12.4711701969306,    0.772699435551961])

# plt.legend(['AKAZE','ORB','SIFT','SURF'], loc='lower left')
# plt.axis([None,None, 0, 13])
# plt.ylabel('computational_time_average (sec)')
# plt.xlabel(' modified images ')
# plt.title('algos_KNN_matching_time')
# plt.show()
# # plt.pause(0.05)
# #plt.savefig('../database/bf_results/knn_matching_time_algos.png')

################################################
#time for getting descriptors knn comparing algos

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


############################################

# similarity percentage comparing between algos _KNN

# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[55.8822719812, 71.1641894623, 86.1820443954, 41.2494103330])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[19.1, 91.0, 65, 97])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[57.80013289, 80.5709513950, 81.8748116254, 52.8376568889])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[15.25934353435, 91.94862396455, 74.87631275951, 34.1364597792])

# plt.legend(['AKAZE','ORB','SIFT','SURF'], loc='lower left')
# plt.axis([None,None, 0, 100])
# plt.ylabel('percentage similarity')
# plt.xlabel(' modified images ')
# plt.title('knn_similarity_percentage')
# plt.show()
# # plt.pause(0.05)
# # plt.savefig('../database/bf_results/knn_matching_time_algos.png')


##############################################

#time for matching algos comparing using FLANN

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

#########################################

#time for getting descriptors FLANN comparing algos

plt.plot(['rotated_45','rotated_90','cropped','blurred'],[13.18976329, 2.969968061, 2.905574723, 2.722227371])
plt.plot(['rotated_45','rotated_90','cropped','blurred'],[5.70571029, 4.193383998, 4.149927938, 2.519072461])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[14.33533716, 3.38424399693807, 3.36949411431948, 3.05343625942866])
# plt.plot(['rotated_45','rotated_90','cropped','blurred'],[6.25990654627482, 4.66048846046130, 4.56513202985128, 2.80168905258179])

plt.legend(['SIFT','SURF'], loc='lower left')
plt.axis([None,None, 0, 15])
plt.ylabel('getting_descriptors_time_average (sec)')
plt.xlabel(' modified images ')
plt.title('algos_BF_FLANN_desc')
plt.show()
# plt.pause(0.05)
# plt.savefig('../database/bf_results/knn_descriptors_time_algos.png')

##############################################

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