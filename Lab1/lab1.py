#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:46:47 2020

@author: yyuan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#%% Read the data

def leg_label_process(data):
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    le = LabelEncoder()
    data = le.fit_transform(data)
    return data

def parse_data(data_path):
    col_name = ['animal_name','hair','feathers','eggs','milk','airborne','aquatic',
            'predator','toothed','backbone','breathes','venomous','fins','legs',
            'tail','domestic','catsize','type']

    df = pd.read_csv("./Data/zoo.data",names = col_name)
    
    df['legs'] = leg_label_process(df["legs"])
    color = df['type']
    name = df.animal_name
    y = pd.get_dummies(df['legs'],prefix ="legs")
    df = pd.concat([df,y],axis=1)
    df = df.drop(columns = ['animal_name','legs','type'])
    return df, color, name

def pca(X, n=2):
    
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n)
    pca.fit(X)
    
    pca_ratio = pca.explained_variance_ratio_
    pca_matrix = pca.transform(X)
    #print(pca.components_)
    return pca_ratio, pca_matrix

def compute_distance(A):
    """
    Compute distance matrix.

    Parameters
    ----------
    A : Array
        N_sample x D_dimension.

    Returns
    -------
    D : Array.
        N_sample x N_sample.

    """
    from sklearn.metrics import pairwise_distances
    
    D = pairwise_distances(A,metric ='l2' )
    
    return D

def mds(X, n_component = 2, matrix_type = "dist", attribute_importance = False):
    if matrix_type not in ["dist","gram","data"]:
        raise ValueError("Wrong type of matrix!")
        pass
    elif matrix_type == "dist":
        # Double centering
        S = double_centering(X)
        
    elif matrix_type == "data":
        # Calculate gram matrix
        n,m =X.shape
        X = X-1/m * X@np.ones((m,1))@np.ones((m,1)).T
        S = X.T @X

    # Gram matrix eigenvalue-decomposition
    eigen_value, eigen_vector =LA.eig(S)
    eigen_value=eigen_value.real
    eigen_vector = eigen_vector.real
    
    # Sort the eigen value and right eigen factor
    arg = np.argsort(-eigen_value)
    eigen_value = eigen_value[arg]
    eigen_vector = eigen_vector[:,arg]
    
    lamb = np.sqrt(np.diag(eigen_value[:n_component]))
    v = eigen_vector[:,0:n_component]
    
    Y = lamb @ v.T
    return Y

def double_centering(D):
    n,m = D.shape
    # print(n)
    DD =np.square(D)
    
    I1 = np.ones((n,1))
    
    S = -0.5*(DD - (1/n)*DD @I1 @I1.T 
             - (1/n)* I1 @I1.T @DD 
             + (1/(n**2))*I1 @I1.T @DD @I1 @I1.T)
    
    return S

def isomap(X, k = 5):
    # Build graph according to euclidean length
    K = isomap_distance(X, k)
    # Compute the shortest graph distance, and square
    from sklearn.utils.graph_shortest_path import graph_shortest_path
    
    G = graph_shortest_path(K)
    #print(np.unique(G, return_counts=True))
    # Double centering
    # G = double_centering(G)
    # MDS
    Y = mds(G)
    
    return Y

def isomap_distance(X, k):
    """
    Generate k-nearest neighborhood distance matrix. 

    Parameters
    ----------
    X : TYPE
        Data matrix. N_sample x D_dimension
    k : TYPE, int
        K-nearest neighborhood. The default is 5.

    Returns
    -------
    Distance matrix. if nertex is connected, then dist(i,j) has distance, else dist(i,j)=0.

    """
    # Calculate pairwise distance
    D = compute_distance(X)
    r,c = D.shape
    
    K = np.zeros((r,c))
    
    #Select K-nearest distance for each point
    for node in range(r):
        # Select k-nearest points for each node
        arg = np.argsort(D[node,:])
        arg = arg[arg != node][:k]
        #print(arg)
        # Assign distance to K matrix
        for i in arg:
            if D[node,i] ==0:
                epsilon = 0.0001
                K[node,i] = D[node,i]+epsilon
                K[i,node] = D[node,i]+epsilon
            else:
                K[node,i] = D[node,i]
                K[i,node] = D[node,i]
        #print(K[node,:])

    return K    

#%%
if __name__ == '__main__':
    
    data_path = "./Data/zoo.data"
    df, color, name = parse_data(data_path)
    
    pca_ratio, pca_matrix = pca(df,2)
    
    #PCA plotting
    plt.figure()
    plt.scatter(pca_matrix[:,0],pca_matrix[:,1],c=color, cmap = 'plasma')
    plt.title("Dimension Reduction with PCA")
    plt.xlabel("PC1({0}%)".format(int(pca_ratio[0]*100)))
    plt.ylabel("PC2({0}%)".format(int(pca_ratio[1]*100)))
    plt.show()
    #MDS
    D = compute_distance(df)
    mds_matrx = mds(df.T,matrix_type="data")
    
    #MDS plotting with data matrix
    plt.figure()
    plt.scatter(mds_matrx[0],mds_matrx[1],c=color, cmap = 'plasma')
    plt.title("Dimension Reduction with MDS(data matrix)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    # MDS plotting with distance matrix
    mds_matrx = mds(D)
    plt.figure()
    plt.scatter(mds_matrx[0],mds_matrx[1],c=color, cmap = 'plasma')
    plt.title("Dimension Reduction with MDS(distance matrix)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    # Isomap plotting
    isomap_matrix = isomap(df, k=20)
    plt.figure()
    plt.scatter(isomap_matrix[0],isomap_matrix[1],c=color, cmap = 'plasma')
    plt.title("Dimension Reduction with Isomap(k=20)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()