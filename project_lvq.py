#!/usr/bin/env python
# coding: utf-8

# In[34]:


import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklvq import GLVQ
import numpy as np
import pandas as pd

names = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'DEATH_EVENT']
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

df.drop(columns=['time'], inplace=True)
y = df[df.columns[-1]]
cut = int(len(df) * .75)
df_train = df[1:cut]
y_train = y[1:cut]
df_test = df[cut:]
y_test = y[cut:]

scaler = StandardScaler()
data = scaler.fit_transform(df_train)

model = GLVQ(
    distance_type = "squared-euclidean",
    activation_type="swish",
    activation_params={"beta":2},
    solver_type="steepest-gradient-descent",
    solver_params={"max_runs": 20, "step_size": 0.1},
)


# In[35]:


model.fit(data, y_train)

predicted = model.predict(df_test)

print(classification_report(y_test, predicted))


# In[36]:


num_prototypes = model.prototypes_.shape[0]
num_features = model.prototypes_.shape[1]

transformed_prototypes = [scaler.inverse_transform(prototype) for prototype in model.prototypes_]


# In[37]:


x = np.arange(1)
width = 0.5

plt.rcParams['figure.figsize'] = [5, 10]
plt.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(nrows=4, ncols=3)
fig.suptitle("Prototype for NHF and HF")

#rects1 = ax.bar(x - width/2, transformed_prototypes[0], width, label='No heart failure')
#rects2 = ax.bar(x + width/2, transformed_prototypes[1], width, label='Heart failure')

i = 0
for row in range(4):
    for col in range(3):
        ax[row][col].set_title(f"{names[i]}")
        ax[row][col].set_xticks([])
        rect1 = ax[row][col].bar(1-width, transformed_prototypes[0][i], label="NHF")
        rect2 = ax[row][col].bar(1+width, transformed_prototypes[1][i], label="HF")
        i+=1

fig.tight_layout()
plt.show()

