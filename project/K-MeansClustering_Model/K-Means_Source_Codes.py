#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Library
import warnings # To hide warnings
import numpy as np  # Data 
import pandas as pd # Dataframe 
import matplotlib.pyplot as plt # For graphics
import seaborn as sns
from sklearn.preprocessing import StandardScaler#For Normalizing data
from sklearn.cluster import KMeans #For clustering
import os                    
import sys    
from sklearn.decomposition import PCA #For Transforming data to PCA

#To drop unneeded columns
def processDataFrame(dataframe):
    new_df = dataframe.drop(['Kind', 'Name'], axis=1)
    return new_df

# To visualize data
def visualizeData (dataframe, title):
    f, ax = plt.subplots(figsize=(7, 5))
    corr = dataframe.corr()
    sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f', linewidths=.05)
    t= f.suptitle(title, fontsize=15)   
    
    
# To preprocess Train data
def processTrainData(dataframe):
    ss = StandardScaler()
    X =ss.fit_transform(dataframe)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    pca_df = pd.DataFrame(data = principalComponents
             , columns = ['PCA1', 'PCA2'])
    return pca_df

#To build K-Mean Model
def initializeModel():
    model = KMeans(n_clusters=3, init='k-means++',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=1099, verbose=0, algorithm='full')
    return model

#To fit model
def fitModel(trainData, model):
    model.fit(trainData)
    return model

#To cluster and predict quality
def predictQuality(trainData, model):
    pred_label = model.predict(trainData)
    centroids = model.cluster_centers_
    inertia = model.n_iter_
    numIter = model.inertia_
    return centroids, pred_label, inertia, numIter

#To save cluster number and predicted quality to a new dataframe
def saveValues(trainData, dataframe, labels, title):
    new_df = dataframe
    new_df['Assigned_Cluster'] = labels
    new_df['PCA1'] = trainData['PCA1']
    new_df['PCA2'] = trainData['PCA2']
    new_df.loc[new_df['Assigned_Cluster']==0,'Predicted_Quality'] = 'High'
    if title == "Picasso":
        new_df.loc[new_df['Assigned_Cluster']==1,'Predicted_Quality'] = 'High'
        new_df.loc[new_df['Assigned_Cluster']==0,'Predicted_Quality'] = 'Medium'
        new_df.loc[new_df['Assigned_Cluster']==2,'Predicted_Quality'] = 'Low'
    else:
        new_df.loc[new_df['Assigned_Cluster']==0,'Predicted_Quality'] = 'High'
        new_df.loc[new_df['Assigned_Cluster']==2,'Predicted_Quality'] = 'Medium'
        new_df.loc[new_df['Assigned_Cluster']==1,'Predicted_Quality'] = 'Low'
    return new_df

#To visualize predicted clusters
def showClusters(new_df, title, centroids):
    fig = plt.figure(figsize = (4,4))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Components 1', fontsize = 10)
    ax.set_ylabel('Principal Components 2', fontsize = 10)
    ax.set_title('K-Means Clustering_' + title, fontsize = 13)
    targets = ['Low', 'Medium', 'High']
    colors = ['c', 'b', 'm']
    for target, color in zip(targets,colors):
        indicesToKeep = new_df['Predicted_Quality'] == target
        ax.scatter(new_df.loc[indicesToKeep, 'PCA1']
               , new_df.loc[indicesToKeep, 'PCA2'], alpha=0.5
               , c = color 
               , s = 30)
        ax.legend(['Low Quality', 'Medium Quality', 'High Quality'])
    #plt.scatter(centroids[:,0] ,centroids[:,1], color='red', s=50, marker ='*')
   
     