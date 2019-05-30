import pickle

def save_clusters(clusters,file1):
    output = open(file1, 'a')
    # writing to a database
    pickle.dump(clusters, output)
    output.close()


def read_clusters(file1):
    pkl_file = open(file1,'rb')
# get zipped object
    print('reading descriptors from a file')
    cluster_obj = pickle.load(pkl_file)

    return cluster_obj