#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random
from typing import List, TypeVar, Tuple

heartData = pd.read_csv('heart.csv', encoding = 'latin1', header = 0)

heartTraining, heartTesting = train_test_split(heartData, test_size = 0.30, random_state = 1, shuffle = True)
knn = KNeighborsClassifier(n_neighbors = 17)
x = heartTraining.iloc[:, :10].values
y = heartTraining.iloc[:, 12].values
knn.fit(x, y)
expected = heartTesting.iloc[:, 12].values
predicted = knn.predict(heartTesting.iloc[:, :10].values)
print(f"Random State = 1")
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print(f"")

heartTraining, heartTesting = train_test_split(heartData, test_size = 0.30, random_state = 2, shuffle = True)
knn = KNeighborsClassifier(n_neighbors = 17)
x = heartTraining.iloc[:, :10].values
y = heartTraining.iloc[:, 12].values
knn.fit(x, y)
expected = heartTesting.iloc[:, 12].values
predicted = knn.predict(heartTesting.iloc[:, :10].values)
print(f"Random State = 2")
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print(f"")

heartTraining, heartTesting = train_test_split(heartData, test_size = 0.30, random_state = 3, shuffle = True)
knn = KNeighborsClassifier(n_neighbors = 17)
x = heartTraining.iloc[:, :10].values
y = heartTraining.iloc[:, 12].values
knn.fit(x, y)
expected = heartTesting.iloc[:, 12].values
predicted = knn.predict(heartTesting.iloc[:, :10].values)
print(f"Random State = 3")
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print(f"")

heartTraining, heartTesting = train_test_split(heartData, test_size = 0.30, random_state = 4, shuffle = True)
knn = KNeighborsClassifier(n_neighbors = 17)
x = heartTraining.iloc[:, :10].values
y = heartTraining.iloc[:, 12].values
knn.fit(x, y)
expected = heartTesting.iloc[:, 12].values
predicted = knn.predict(heartTesting.iloc[:, :10].values)
print(f"Random State = 4")
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[ ]:




