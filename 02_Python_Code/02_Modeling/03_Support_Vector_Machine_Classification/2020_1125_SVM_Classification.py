#!/usr/bin/env python
# coding: utf-8

# # Classifying Pulsars from the High Time Resolution Universe Survey (HTRU2) - Support Vector Machine (SVM) Classification

# ## Overview & Citation

# In this code notebook, we attempt to classify pulsars from the High Time Resolution Universe Survey, South (HTRU2) dataset using support vector machine (SVM) classification. The dataset was retrieved from the UC Irvine Machine Learning Repository at the following link: https://archive.ics.uci.edu/ml/datasets/HTRU2#.
# 
# The dataset was donated to the UCI Repository by Dr. Robert Lyon of The University of Manchester, United Kingdom. The two papers requested for citation in the description are listed below:
# 
# * R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
# * R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1.

# ## Import the Relevant Libraries

# In[12]:


# Data Manipulation
import pandas as pd
import numpy as np

# Modeling & Evaluation
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report


# ## Import & Check the Data

# In[2]:


df = pd.read_csv('2020_1125_Pulsar_Data.csv')
pulsar_data = df.copy()


# In[3]:


pulsar_data.head()


# ## Train Test Split

# In[4]:


X = pulsar_data.drop('Class',axis=1)
y = pulsar_data['Class']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# ## SVM with Default Parameters

# ### Construct and Test the Model

# In[6]:


svm_model = SVC()
svm_model.fit(X_train,y_train)


# In[7]:


y_pred = svm_model.predict(X_test)


# ### Model Evaluation

# In[8]:


confusion = confusion_matrix(y_test,y_pred)
print(f'CONFUSION MATRIX:\n\n{confusion[0][0]}\t{confusion[0][1]}\n{confusion[1][0]}\t{confusion[1][1]}')


# In[9]:


print(f'CLASSIFICATION REPORT:\n\n{classification_report(y_test,y_pred)}')


# The dataset contains a total of 1,639 actual pulsars out of 16,259 instances in the dataset (approximately 10%). This means that we have an unbalanced classification problem, and accuracy is not a good metric. Therefore, the most important metrics for predicting a pulsar with this model are:
# * Precision = 0.94
# * Recall = 0.76
# * F1-Score = 0.84

# ## Optimizing Performance

# ### Run a Cross-Validation Grid Search

# Let's try to improve the evaluation metrics by running a grid search to optimize the C and gamma parameters of the support vector classifier below:

# In[13]:


grid_parameters = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[16]:


grid = GridSearchCV(SVC(),grid_parameters,refit=True,verbose=3)


# In[17]:


# Note - this will take a few minutes to run
grid.fit(X_train,y_train)


# In[18]:


# Find the best parameters:
grid.best_params_


# In[19]:


# Find the best estimator:
grid.best_estimator_


# ### Run the Optimized Model

# In[21]:


svm_optimized = SVC(C=1000, gamma=0.0001)
svm_optimized.fit(X_train,y_train)
y_pred_optimized = svm_optimized.predict(X_test)


# ### Model Evaluation

# In[23]:


confusion = confusion_matrix(y_test,y_pred_optimized)
print(f'CONFUSION MATRIX:\n\n{confusion[0][0]}\t{confusion[0][1]}\n{confusion[1][0]}\t{confusion[1][1]}')


# In[24]:


print(f'CLASSIFICATION REPORT:\n\n{classification_report(y_test,y_pred_optimized)}')


# ## Conclusions

# Running the cross-validation grid search resulted in the following improvements:
# * Accuracy increased from 0.97 to 0.98
# * Recall increased from 0.76 to 0.83
# * F1-Score increased from 0.84 to 0.88
# 
# Let's save the results of the optimized model for future reference:

# In[25]:


with open("2020_1125_SVM_Results.csv","w") as file:
    file.write('Model,Accuracy,Precision,Recall,F1-Score\n')
    file.write('SVM,0.98,0.94,0.83,0.88\n')

