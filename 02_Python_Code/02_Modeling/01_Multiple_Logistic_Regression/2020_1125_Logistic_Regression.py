#!/usr/bin/env python
# coding: utf-8

# # Classifying Pulsars from the High Time Resolution Universe Survey (HTRU2) - Logistic Regression

# ## Overview & Citation

# In this code notebook, we attempt to classify pulsars from the High Time Resolution Universe Survey, South (HTRU2) dataset using logistic regression. The dataset was retrieved from the UC Irvine Machine Learning Repository at the following link: https://archive.ics.uci.edu/ml/datasets/HTRU2#.
# 
# The dataset was donated to the UCI Repository by Dr. Robert Lyon of The University of Manchester, United Kingdom. The two papers requested for citation in the description are listed below:
# 
# * R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
# * R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1.

# ## Import the Relevant Libraries

# In[34]:


# Data Manipulation
import pandas as pd
import numpy as np

# Modeling & Evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report


# ## Import & Check the Data

# In[3]:


df = pd.read_csv('2020_1125_Pulsar_Data.csv')
pulsar_data = df.copy()


# In[4]:


pulsar_data.head()


# ## Train Test Split

# In[6]:


X = pulsar_data.drop('Class',axis=1)
y = pulsar_data['Class']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[10]:


logmodel = LogisticRegression(max_iter=200) # Max iterations = 200 since default 100 does not work
logmodel.fit(X_train,y_train)


# In[12]:


y_pred = logmodel.predict(X_test)


# ## Model Evaluation

# In[29]:


confusion = confusion_matrix(y_test,y_pred)
print(f'CONFUSION MATRIX:\n\n{confusion[0][0]}\t{confusion[0][1]}\n{confusion[1][0]}\t{confusion[1][1]}')


# In[28]:


print(f'CLASSIFICATION REPORT:\n\n{classification_report(y_test,y_pred)}')


# The dataset contains a total of 1,639 actual pulsars out of 16,259 instances in the dataset (approximately 10%). This means that we have an unbalanced classification problem, and accuracy is not a good metric. Therefore, the most important metrics for predicting a pulsar with this model are:
# * Precision = 0.94
# * Recall = 0.81
# * F1-Score = 0.87
# 
# Let's save this data to a .csv file for future comparison with the other classification models:

# In[33]:


with open("2020_1125_Logistic_Results.csv","w") as file:
    file.write('Model,Accuracy,Precision,Recall,F1-Score\n')
    file.write('Logistic,0.98,0.94,0.81,0.87\n')

