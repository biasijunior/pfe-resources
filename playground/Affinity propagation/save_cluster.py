import pickle

def save_clusters(clusters,file1):
    output = open(file1, 'wb')
    # writing to a database
    pickle.dump(clusters, output)
    output.close()

def save_matrix(matrix,file1):
    output = open(file1, 'wb')
    print("saving a matrix...")
    # writing to a database
    pickle.dump(matrix, output)
    output.close()

def read_clusters(file1):
    pkl_file = open(file1,'rb')
    print('reading descriptors from a file')
    cluster_obj = pickle.load(pkl_file)
    pkl_file.close()
    return cluster_obj

def get_matrix(file1):
    pkl_file = open(file1,'rb')
    print('get a matrix of descriptors from a file')
    matrix = pickle.load(pkl_file)
    pkl_file.close()
    return matrix