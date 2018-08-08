from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
import mnist_reader
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import defaultdict
from scipy.spatial import distance
import os
import random as rand

def float_compare(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def centeroid_step(clusters,text):
    centeroids=[]
    for i in clusters:
        if text:
            nparr = np.array([j[0] for j in clusters[i]])
            mean1 = np.mean(nparr,axis=0)
            
            sim = np.matmul(nparr, mean1)
            ind = np.unravel_index(np.argmin(sim, axis=None), sim.shape)
           
            centeroids.append(nparr[ind[0]])
        else:    
            centeroids.append(np.mean(np.array([j[0] for j in clusters[i]]),axis=0))

    return centeroids

def clustering_step(k,centeroids,data,label,cosine):
    clusters={}
    for i in range(k):
        clusters[i]=[]    
    
    if(cosine):
        sim = cosine_distances(data,centeroids)
    else:
        sim = euclidean_distances(data,np.array(centeroids))
    
    c=0
    for i in sim:
        ind = np.unravel_index(np.argmin(i, axis=None), i.shape)
        clusters[ind[0]].append((data[c],label[c]))
        c+=1
    return clusters

def Kmeans(k, data, label, cosine,text):
    centeroids=[]
    clusters={}
    
    for i in range(k):
        clusters[i]=[]
        
    for i in range(k):
        centeroids.append(data[rand.randint(0,len(label))])
            
    prev_centeroids = centeroids
    prev_clusters = clusters
    
    ctr = 0
    ch=0
    while(True):
        clusters = clustering_step(k,centeroids,data,label,cosine)
        centeroids = centeroid_step(clusters,text)
        
        pt=True
        #eud=[]
        for i in range(len(prev_centeroids)):
            #eud.append(distance.euclidean(prev_centeroids[i],centeroids[i]))
            for j in range(len(prev_centeroids[i])):
                if(not float_compare(prev_centeroids[i][j],centeroids[i][j])):
                    pt=False

        if(pt):
            break
        
        ch=0
        for i in clusters:
            ch+=abs(len(prev_clusters[i])-len(clusters[i]))
        if(ctr%10==0):
            print "round: ",ctr," number of shuffles: ",ch
        
        ctr+=1
        prev_centeroids = centeroids
        prev_clusters = clusters

    ch=0
    for i in clusters:
        ch+=abs(len(prev_clusters[i])-len(clusters[i]))
    if(ctr%10!=0):
        print "round: ",ctr," number of shuffles: ",ch

    return centeroids,clusters


def performance_calculator(cluster,categories):
    confusion_matrix=[]
    for i in cluster:
        countDict=defaultdict(int)
        for j in cluster[i]:
            countDict[j[1]]+=1
    
        confusion_matrix.append([countDict[i] for i in range(categories)])
    
    
    numerator = 0
    denominator = 0
    for i in confusion_matrix:
        numerator+=max(i)
        denominator+=sum(i)
    
    purity = float(numerator)/float(denominator)
    
    gini_indexes=[]
    total=0
    for i in confusion_matrix:
        gi=1
        for j in i:
            gi-=((float(j)/float(sum(i)))**2)
        gini_indexes.append(gi*sum(i))
        total+=sum(i)
        
    gini = float(sum(gini_indexes))/float(total)
    
    return (purity,gini,confusion_matrix)


def runKmeans(k,X,y,nl,cosine=False,Text=False):
    centeroids,clusters = Kmeans(k,X,y,cosine,Text)
    purity,gini,confusion_matrix = performance_calculator(clusters,nl)
    
    return (purity,gini)
