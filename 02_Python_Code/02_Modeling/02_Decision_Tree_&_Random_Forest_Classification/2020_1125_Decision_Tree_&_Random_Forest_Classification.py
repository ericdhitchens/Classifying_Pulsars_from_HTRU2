#!/usr/bin/env python
# coding: utf-8

# # Classifying Pulsars from the High Time Resolution Universe Survey (HTRU2) - Decision Tree & Random Forest Classification

# ## Overview & Citation

# In this code notebook, we attempt to classify pulsars from the High Time Resolution Universe Survey, South (HTRU2) dataset using decision tree and random forest classification. The dataset was retrieved from the UC Irvine Machine Learning Repository at the following link: https://archive.ics.uci.edu/ml/datasets/HTRU2#.
# 
# The dataset was donated to the UCI Repository by Dr. Robert Lyon of The University of Manchester, United Kingdom. The two papers requested for citation in the description are listed below:
# 
# * R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
# * R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1.

# ## Import the Relevant Libraries

# In[1]:


# Data Manipulation
import pandas as pd
import numpy as np

# Modeling & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report


# ## Import & Check the Data

# In[2]:


df = pd.read_csv('2020_1125_Pulsar_Data.csv')
pulsar_data = df.copy()


# In[3]:


pulsar_data.head()


# ## Train Test Split

# The following train test split will be used for both the decision tree and random forest classifications below:

# In[4]:


X = pulsar_data.drop('Class',axis=1)
y = pulsar_data['Class']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# ## Decision Tree Classification

# ### Build and Test the Model

# In[6]:


tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)


# In[7]:


y_pred = tree.predict(X_test)


# ### Model Evaluation

# In[8]:


confusion = confusion_matrix(y_test,y_pred)
print(f'CONFUSION MATRIX:\n\n{confusion[0][0]}\t{confusion[0][1]}\n{confusion[1][0]}\t{confusion[1][1]}')


# In[9]:


print(f'CLASSIFICATION REPORT:\n\n{classification_report(y_test,y_pred)}')


# The dataset contains a total of 1,639 actual pulsars out of 16,259 instances in the dataset (approximately 10%). This means that we have an unbalanced classification problem, and accuracy is not a good metric. Therefore, the most important metrics for predicting a pulsar with this model are:
# * Precision = 0.84
# * Recall = 0.86
# * F1-Score = 0.85
# 
# Let's save this data to a .csv file for future comparison with the other classification models:

# In[10]:


with open("2020_1125_Decision_Tree_Results.csv","w") as file:
    file.write('Model,Accuracy,Precision,Recall,F1-Score\n')
    file.write('Decision_Tree,0.97,0.84,0.86,0.85\n')


# ## Random Forest Classification

# ### Build and Test the Model

# In[11]:


forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train,y_train)


# In[12]:


y_pred = forest.predict(X_test)


# ### Model Evaluation

# In[13]:


confusion = confusion_matrix(y_test,y_pred)
print(f'CONFUSION MATRIX:\n\n{confusion[0][0]}\t{confusion[0][1]}\n{confusion[1][0]}\t{confusion[1][1]}')


# In[14]:


print(f'CLASSIFICATION REPORT:\n\n{classification_report(y_test,y_pred)}')


# We see that the random forest classifier has performed significantly better than the decision tree classifier in terms of precision and f1-score. Let's see if we can improve the performance of our random forest by experimenting with the number of estimators in the random forest.

# ### Improving Performance

# In[15]:


# Test the data with random forest classifiers of variable n_estimators.
# Please note that this may take a few minutes to run.

for i in range(100,1600,100):
    forest_loop = RandomForestClassifier(n_estimators=i)
    forest_loop.fit(X_train,y_train)
    y_pred_loop = forest_loop.predict(X_test)
    
    print(f'CLASSIFICATION REPORT FOR {i} ESTIMATORS:\n\n{classification_report(y_test,y_pred_loop)}\n\n')


# The best random forest model used 1000 estimators and yielded the following metrics:
# * Accuracy = 0.98
# * Precision = 0.94
# * Recall = 0.84
# * F1-Score = 0.88
# 
# Let's save this in a .csv file for future reference:

# In[16]:


with open("2020_1125_Random_Forest_Results.csv","w") as file:
    file.write('Model,Accuracy,Precision,Recall,F1-Score\n')
    file.write('Random_Forest,0.98,0.94,0.84,0.88\n')


# ## Conclusions

# We conclude that the random forest classifier performed better than the decision tree classifier. We also conclude that increasing the number of estimators (number of trees in the random forest) does not appreciably improve the predictive power of the random forest, at least when tested over range(100,1600,100). 
# 
