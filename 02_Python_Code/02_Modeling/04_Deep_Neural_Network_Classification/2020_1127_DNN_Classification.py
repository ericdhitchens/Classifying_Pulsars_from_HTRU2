#!/usr/bin/env python
# coding: utf-8

# # Classifying Pulsars from the High Time Resolution Universe Survey (HTRU2) - Deep Neural Network (DNN) Classification

# ## Overview & Citation

# In this code notebook, we attempt to classify pulsars from the High Time Resolution Universe Survey, South (HTRU2) dataset using deep neural network (DNN) classification. The dataset was retrieved from the UC Irvine Machine Learning Repository at the following link: https://archive.ics.uci.edu/ml/datasets/HTRU2#.
# 
# The dataset was donated to the UCI Repository by Dr. Robert Lyon of The University of Manchester, United Kingdom. The two papers requested for citation in the description are listed below:
# 
# * R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
# * R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1.

# ## Import the Relevant Libraries

# In[108]:


# Data Manipulation
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ANN Modeling in TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

# Model Evaluation
from sklearn.metrics import classification_report,confusion_matrix


# ## Data Preprocessing

# ### Import & Check the Data

# In[109]:


df = pd.read_csv('2020_1125_Pulsar_Data.csv')
pulsar_data = df.copy()


# In[110]:


pulsar_data.head()


# ### Train Test Split

# In[111]:


X = pulsar_data.drop('Class',axis=1)
y = pulsar_data['Class']


# In[112]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# ### Scale the Data

# In[113]:


scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Model 1 - DNN with 2 Hidden Layers

# ### Construct the Deep Neural Network

# In[115]:


# Determine number of starting nodes by finding the shape of X_train
X_train.shape


# In[133]:


model = Sequential()

# Input Layer
model.add(Dense(8,activation='relu')) # All layers utilize rectified linear units (relu)

# Hidden Layers
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))

# Output Layer (Sigmoid for Binary Classification)
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


# ### Train the Model on the Test Data

# In[134]:


model.fit(x=X_train, y=y_train, epochs=200, validation_data=(X_test, y_test), verbose=1)


# ### Visualize the Loss Function

# In[135]:


losses = pd.DataFrame(model.history.history)
losses.plot()


# Our validation loss remains relatively stable with the actual loss, so overfitting is minimal.

# ### Test the Model

# In[136]:


#y_pred = np.argmax(model.predict(X_test),axis=-1)
y_pred = model.predict_classes(X_test)


# ### Model Evaluation

# In[137]:


confusion = confusion_matrix(y_test,y_pred)
print(f'CONFUSION MATRIX:\n\n{confusion[0][0]}\t{confusion[0][1]}\n{confusion[1][0]}\t{confusion[1][1]}')


# In[139]:


print(f"CLASSIFICATION REPORT:\n\n{classification_report(y_test,y_pred)}")


# ## Model Optimization

# ### Iterating Through the Models

# Let's experiment with the number of hidden layers in our network. We'll iterate from 2 to 50 hidden layers, each containing 8 units

# In[145]:


results = []
for i in range(2,51):
    model_loop = Sequential()
    
    # Input and hidden layers
    for j in range(0,(i+1)):
        model_loop.add(Dense(8,activation='relu'))
    
    # Output layer
    model_loop.add(Dense(1,activation='sigmoid'))

    # Compile layers
    model_loop.compile(loss='binary_crossentropy', optimizer='adam')
    
    # We will reduce epochs to 100 to reduce run time. 
    # 100 was chosen based on previous loss function visualization.
    model_loop.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=100,verbose=0)
    
    # Model evaluation
    predictions_loop = model_loop.predict_classes(X_test)
    
    # Calculate statistics for each iteration
    confusion = confusion_matrix(y_test,predictions_loop)
    
    tp = confusion[1][1]
    tn = confusion[0][0]
    fp = confusion[0][1]
    fn = confusion[1][0]
    
    total = tp+tn+fp+fn
    accuracy = (tp+tn)/total
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    
    # Append results to the list for future reference
    results.append([i,accuracy,precision,recall,f1_score])
    
    # Print classification report after each iteration
    print(f"CLASSIFICATION REPORT FOR {i} HIDDEN LAYERS:\n\n{classification_report(y_test,predictions_loop)}")


# ### Cleaning the Results

# Display the results in a pandas dataframe

# In[152]:


results_df = pd.DataFrame(columns=['Hidden_Layers','Accuracy','Precision','Recall','F1-Score'],data=results)
results_df


# There appear to be many null values. Let's remove them.

# In[155]:


cleaned_results = results_df.dropna()
cleaned_results


# ### Visualizing the Iterations

# Let's visualize the performance by number of hidden layers.

# In[161]:


# Visualizing Accuracy
plt.scatter(x=cleaned_results['Hidden_Layers'],y=cleaned_results['Accuracy'])


# In[163]:


# Visualizing Precision
plt.scatter(x=cleaned_results['Hidden_Layers'],y=cleaned_results['Precision'])


# In[164]:


# Visualizing Recall
plt.scatter(x=cleaned_results['Hidden_Layers'],y=cleaned_results['Recall'])


# In[165]:


# Visualizing F1-Score
plt.scatter(x=cleaned_results['Hidden_Layers'],y=cleaned_results['F1-Score'])


# ### Analyzing the Results

# In[162]:


cleaned_results.describe()


# In[181]:


# Display Top 5 Models by Accuracy
cleaned_results.sort_values(by='Accuracy',ascending=False).drop(['Precision','Recall','F1-Score'],axis=1)[:5]


# In[178]:


# Display Top 5 Models by Precision
cleaned_results.sort_values(by='Precision',ascending=False).drop(['Accuracy','Recall','F1-Score'],axis=1)[:5]


# In[183]:


# Display Top 5 Models by Recall
cleaned_results.sort_values(by='Recall',ascending=False).drop(['Accuracy','Precision','F1-Score'],axis=1)[:5]


# In[182]:


# Display Top 5 Models by F1-Score
cleaned_results.sort_values(by='F1-Score',ascending=False).drop(['Accuracy','Precision','Recall'],axis=1)[:5]


# ## Conclusions

# The best performing models for each category below (number of hidden layers in parentheses) were:
# * Accuracy = 0.981 from Model(38)
# * Precision = 0.959 from Model(35)
# * Recall = 0.869 from Model(31)
# * F1-Score = 0.890 from Model(38)
# 
# The objective of this project is to find a model that can correctly identify pulsars, which account for only approximately 10% of the dataset. Therefore, the most important metric is recall for this exercise. The model with the best recall had 31 hidden layers. Its evaluation metrics are saved below for future comparison with the other models:

# In[1]:


with open("2020_1127_DNN_Results.csv","w") as file:
    file.write('Model,Accuracy,Precision,Recall,F1-Score\n')
    file.write('DNN,0.98,0.90,0.87,0.88\n')

