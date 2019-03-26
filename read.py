import pprint
import cPickle as pickle
import base64

# pkl_file = open('images.csv', 'rb')

# data1 = pickle.load(pkl_file)
# pprint.pprint(data1)

# pkl_file.close()


with open('images.csv', 'r') as csv_file:
    for line in csv_file:
        line = line.strip('\n')
        b64_str = line.split('|')[0]                    # take the pickled obj
        obj = pickle.loads(base64.b64decode(b64_str))
