#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#read dataset
dataset=pd.read_csv('E:\Project\Mall_Customers.csv')
dataset.shape
dataset.head()

x=dataset.iloc[:,[-2,-1]].values
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("no of clusters")
plt.ylabel("wcss")
plt.show()    
kmeans=KMeans(n_clusters=5)
y_means=kmeans.fit_predict(x)

plt.scatter(x[y_means==0,0],x[y_means==0,1],color='red',s=100,label='cluster 1') 
plt.scatter(x[y_means==1,0],x[y_means==1,1],color='blue',s=100,label='cluster 2') 
plt.scatter(x[y_means==2,0],x[y_means==2,1],color='orange',s=100,label='cluster 3') 
plt.scatter(x[y_means==3,0],x[y_means==3,1],color='green',s=100,label='cluster 4') 
plt.scatter(x[y_means==4,0],x[y_means==4,1],color='magenta',s=100,label='cluster 5') 
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,color='yellow',label='centroid')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()   