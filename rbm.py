import numpy as np


int32 = np.dtype(np.int32)
int32 = int32.newbyteorder('B')
ubyte = np.dtype(np.ubyte) 
ubyte = ubyte.newbyteorder('B')

def mnist():

    X_train, Y_train, X_test, Y_test = [0,0,0,0]

    #Read training data
    with open("train-images-idx3-ubyte", "r") as f:

        magic_number = np.fromfile(f, dtype=int32, count=1)
        n = np.fromfile(f, dtype=int32, count=1)
        shape = np.fromfile(f, dtype=int32, count=2)
        X_train = np.fromfile(f, dtype=ubyte, count=shape[0] * shape[1] * n).reshape(n, shape[0], shape[1])

    #Read training labels
    with open("train-labels-idx1-ubyte", "r") as f:

        magic_number = np.fromfile(f, dtype=int32, count=1)
        n = np.fromfile(f, dtype=int32, count=1)
        Y_train = np.fromfile(f, dtype=ubyte, count=n).reshape(n, 1)




    #Read training data
    with open("t10k-images-idx3-ubyte", "r") as f:

        magic_number = np.fromfile(f, dtype=int32, count=1)
        n = np.fromfile(f, dtype=int32, count=1)
        shape = np.fromfile(f, dtype=int32, count=2)
        X_test = np.fromfile(f, dtype=ubyte, count=shape[0] * shape[1] * n).reshape(n, shape[0], shape[1])

    #Read training labels
    with open("t10k-labels-idx1-ubyte", "r") as f:

        magic_number = np.fromfile(f, dtype=int32, count=1)
        n = np.fromfile(f, dtype=int32, count=1)
        Y_test = np.fromfile(f, dtype=ubyte, count=n).reshape(n, 1)


    return X_train, Y_train, X_test, Y_test



#Read mnist data
X_train, Y_train, X_test, Y_test = mnist()







