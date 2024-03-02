<h1 align="center">Heart-Failure-Data-Analysis</h1>

## Introduction
Cardiovascular Failure
This project investigates the Heart Failure Prediction dataset from Kaggle and evaluates the ability of the k-Nearest Neighbors, Multiple Linear Regression, Random Forests, and Learning Vector Quantization models to predict heart failure outcomes. This report introduces the dataset and details, the data analysis performed, and the resulting figures, results, interpretations, conclusions, and potential for improvement for each model.
## The Dataset
The Heart Failure Prediction dataset consists of twelve clinical factors that attempt to explain and predict mortality as a result of heart failure. These predictors include:

* Age
* Anaemia, the decrease of red blood cells or hemoglobin
* Creatinine Phosphokinase, the level of the CPK enzyme in the blood (mcg/L), Diabetes
* Ejection Fraction, the percentage of blood leaving the heart at each contraction High Blood Pressure
* Platelets
* Serum Creatinine
* Serum Sodium
* Sex
* Smoking
  
The Anaemia, Diabetes, High Blood Pressure, Sex, and Smoking factors are represented using indicator variables where the value of 0 indicates the absence of this variable, while 1 indicates this factor is present for a particular patient.
The outcome, or response, variables for this dataset are Time and Death Event, as this dataset is a time-to-event dataset; the time variable describes the follow-up period for the Death Event result, which is either value 0 or 1. The value 0 represents censorship or no death, while 1 indicates mortality as a result of heart failure. The Death Event variable’s categorical nature leads to its representation as an indicator variable with two values, meaning this dataset represents an example of binary classification.
