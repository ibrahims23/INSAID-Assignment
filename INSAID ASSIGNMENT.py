#!/usr/bin/env python
# coding: utf-8

# # INSAID DATASCIENCE INTERNSHIP ASSIGNMENT SUBMITTED BY IBRAHIM SAYED 

# In[19]:


#Intially we import the necessary libraries pandas and numpy.

import pandas as pd 
import numpy as np


# In[46]:


#This reads the dataset from a CSV file named 'Fraud.csv' located at the specified path. 
#The r before the path specifies that the string is a raw string and should be interpreted as such. The dataset is then stored in a Pandas dataframe named data.
data = pd.read_csv(r'C:\Users\ibrah\Downloads\Fraud.csv')


# In[21]:


# displays the first rows of the dataset.
data.head()


# In[47]:


#displays the last rows of the dataset 
data.tail()


# In[24]:


#gives information about the dataframe
data.info()


# In[48]:


#the shape of the data 
data.shape


# In[26]:


#here are the counts of missing values of each column
data.isna().sum()


# In[27]:


#gives the count of the unique values this one is for isFraud
data.isFraud.value_counts()


# In[28]:


#and this is one is for isFlaggedFraud
data.isFlaggedFraud.value_counts()


# In[29]:


#I have dropped nameOrig and nameDest column as it was not needed for the fraud detection model
data=data.drop(['nameOrig','nameDest'],axis=1)


# In[30]:


#now the time to use preprocessing begins because what we have is categorical data
# we have to covert the categoricald data into numerivcals so label encoder will do the needful
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()
data['type']=label_encoder.fit_transform(data['type'])


# In[31]:


#This code separates the features and the target variable from the dataset.
#X contains all the columns except the 'isFraud' column, while y contains only the 'isFraud' column.
X,y=data.loc[:,data.columns!='isFraud'],data['isFraud']


# In[37]:


#i have kept the train and test_size for 60-40 respectively  
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.40,random_state=42)


# In[38]:


# Standard Scaler is done to ensure that all features have the same scale and that no feature dominates the others in the model.
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[39]:


#I have used two machine learning models from the scikit-learn library: Gaussian Naive Bayes and Logistic Regression.
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics 


# In[40]:


gnb = GaussianNB() 


# In[41]:


gnb.fit(X_train,y_train)


# In[42]:


#The predict() method is then used to make predictions on the test data X_test, and the predictions are stored in y_pred.
y_pred=gnb.predict(X_test)
print("The Accuracy =",metrics.accuracy_score(y_test,y_pred))


# In[44]:


#The accuracy of the Logistic Regression model is also calculated using the accuracy_score() function
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)


# In[45]:


print("The Accuracy = ",metrics.accuracy_score(y_test,y_pred))


# 1)Data cleaning including missing values, outliers and multi-collinearity:
# 
# Ans) This code show cases Data cleaning is a crucial step in data preprocessing that involves identifying and correcting errors and inconsistencies in the data to improve its quality and accuracy.Missing values refer to the absence of data in one or more fields in a dataset.Outliers are data points that are significantly different from other data points in the dataset.
# 
# 
# 2) Describe your fraud detection model in elaboration:
# 
# Ans)The models that i have used is for training and evaluating - Gaussian Naive Bayes and Logistic Regression. Both models are commonly used for binary classification tasks like fraud detection. The data is split into training and test sets, scaled using StandardScaler, and encoded using LabelEncoder. The models are evaluated using the accuracy_score metric.
# 
# 3) How did you select variables to be included in the model?
# 
# Ans) The code drops two columns "nameOrig" and "nameDest"
# 
# 4) Demonstrate the performance of the model by using the best set of tools:
# 
# Ans)The code uses the accuracy_score metric to evaluate the performance of the models. 
# 
# 5) What are the key factors that predict fraudulent customer?
# 
# Ans) Feature importance or coefficients from the models could be used to determine which features have the most predictive power.
# 
# 6) Do these factors make sense? If yes, how? If not, How not?
# 
# Ans) In my opinion it's important to note that some features may not have a direct or obvious relationship with fraud, but could still be useful in predicting fraudulent transactions when combined with other features.
# 
# 7) What kind of prevention should be adopted while the company updates its infrastructure?
# 
# Ans) Preventing fraud requires a multi-faceted approach that includes both technical and non-technical solutions. Some possible technical solutions could include implementing real-time monitoring and analysis of transaction data, using machine learning models to flag potentially fraudulent transactions, and implementing two-factor authentication and other security measures to prevent unauthorized access to customer accounts. Non-technical solutions could include educating customers about the importance of strong passwords and other security practices, and training employees to recognize and report suspicious activity.
# 
# 8) Assuming these actions have been implemented, how would you determine if they work?
# 
# 
# Ans)To determine if these actions are effective, the company could track metrics like the number of fraudulent transactions detected, the false positive rate (i.e. the rate at which legitimate transactions are flagged as fraudulent), and the overall financial impact of fraud on the company. If these metrics improve over time, it could be an indication that the company's fraud prevention measures are working. Additionally, the company could conduct regular audits and assessments of their fraud prevention program to identify areas for improvement.

# In[ ]:




