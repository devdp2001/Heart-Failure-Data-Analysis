#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.drop(columns = 'time', inplace = True)
df.head()


# In[2]:


x, y = df.drop(["DEATH_EVENT"],axis = 1), df["DEATH_EVENT"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

model = RandomForestClassifier(max_features = 0.5, max_depth = 5, random_state = 1)

model.fit(x_train, y_train)
predicted = model.predict(x_test)
print(classification_report(y_test, predicted))


# In[ ]:
