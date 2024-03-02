#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import csv as csv
import matplotlib.pyplot as plt
import math

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures


# In[3]:


hf = pd.read_csv("heart_failure_clinical_records_dataset.csv")


# In[4]:


# Data for set when death due to heart failure occurs
hfDead = hf[hf['DEATH_EVENT'] == 1]


# In[5]:


# Data for set when death due to heart failure does not occur
hfAlive = hf[hf['DEATH_EVENT'] == 0]


# In[6]:


# Histograms for each of the clinical factors and response variables (time and DEATH_EVENT)
hf.hist(alpha=0.8, figsize=(20, 10))
plt.tight_layout()


# In[17]:


# Histograms for each of the clinical factors and response variables (time and DEATH_EVENT) for patients who did not die
hfAlive.hist(alpha=0.8, figsize=(20, 10))
plt.tight_layout()


# In[18]:


# Histograms for each of the clinical factors and response variables (time and DEATH_EVENT) for patients who died
hfDead.hist(alpha=0.8, figsize=(20, 10))
plt.tight_layout()


# In[19]:


print("Correlation between input features and response variable (DEATH_EVENT)")
hf.corr().iloc[12]


# In[20]:


condensed = hf.drop(["anaemia","creatinine_phosphokinase","diabetes","high_blood_pressure","platelets","sex","smoking","time"],1)
condensed.corr()


# In[10]:


print("The four input features that exhibit the greatest correlation with the response variable, DEATH_EVENT, are Serum_Creatanine, Serum_Sodium, Ejection_Fraction, and Age.\n")
hf.plot.scatter("DEATH_EVENT", "serum_creatinine")
hf.plot.scatter("DEATH_EVENT", "serum_sodium")
hf.plot.scatter("DEATH_EVENT", "ejection_fraction")
hf.plot.scatter("DEATH_EVENT", "age")


# In[11]:


# Dataset using condensed set of predictors, based on correlation with DEATH_EVENT response variable
x = hf.drop(["anaemia","creatinine_phosphokinase","diabetes","high_blood_pressure","platelets","sex","smoking","time","DEATH_EVENT"],1)


# In[12]:


# Response values
y = hf["DEATH_EVENT"]


# In[13]:


# Splitting of dataset into training and testing sets (30/70 split test to train)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[14]:


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)


# In[15]:


# The intercept:
print('Intercept: \n', regr.intercept_)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# The coefficient of determination, r2 for model: 1 is perfect prediction
print('R-Squared: %f'
      % regr.score(x_test, y_test))

print('R: %.5f'
      % 0.33093349439)

