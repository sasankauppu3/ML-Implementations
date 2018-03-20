import numpy as np
import h5py

def saveNumpyArray(data,fname):
    h5f = h5py.File(fname, 'w')
    h5f.create_dataset('dataset_1', data)
    h5f.close()
    return

def retrieveNumpyArray(fname):
    h5f = h5py.File(fname,'r')
    data = h5f['dataset_1'][:]
    h5f.close()
    return data

def sq2ind(d,i,j):
    return d*(d-1)/2 - (d-i)*(d-i-1)/2 + j - i - 1

def generateSqIndexes(d,i,tl):
    indices=[]
    for l in range(0,i):
        if l<tl:
            indices.append((l,sq2ind(d,l,i)))

    for l in range(i+1,d):
        if l<tl:
            indices.append((l,sq2ind(d,i,l)))

    return indices
