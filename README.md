# An_Exploratory_Study_of_Cardiovascular_Disease_Prediction_using_Decision_trees_vs_Random_forest

## The idea is to use decision trees and random forest to predict any cardiovascular disease given the patient's history, and compare the results for robustness
####  Cardiovascular disease (CVDs) is the number one cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of five CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management.
This dataset contains 11 features that can be used to predict possible heart disease.
 
Dataset : https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

### Features (Patient history needed on the following parameters):
- Age: age of the patient [years]
- Sex: sex of the patient [M: Male, F: Female]
- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: resting blood pressure [mm Hg]
- Cholesterol: serum cholesterol [mm/dl]
- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), - LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
- Oldpeak: oldpeak = ST [Numeric value measured in depression]
- ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]

Note : one-hot-encoding scheme is utilized on Sex, ChestPainType, RestingECG, ExerciseAngina, and ST_Slope to breakdown features to increase granularity of the data. 

### Test and train data split
85% data is used for training, and 15% data is used for testing.

### Parameters for decision tree and random forest models
 The parameters for decision tree model and random forest model were selected based on the observations from the various graphs such as accuracy Vs max_depth, accuracy Vs min_splits, accuracy Vs n_estimators, for both training set and validation set, in order to ` avoid overfitting or underfitting `

![Figure_1](https://user-images.githubusercontent.com/40464435/224136641-e383d21e-4e46-46f8-bde3-d491e9b9fd59.png)
![Figure_2](https://user-images.githubusercontent.com/40464435/224136638-c67994bf-c363-4bb6-9a20-212fb524e17b.png)
![Figure_3](https://user-images.githubusercontent.com/40464435/224136633-86a508db-d3c6-4fef-8529-d5558386cdc3.png)

## Results
| model |training accuracy| test accuracy |
| ---------------|----------------|--------|
|  Decision tree   | 85.83%  |86.41%|
| Random forest (tree ensemble)  |  94.28%  |  89.13%

### Clearly a tree ensemble or random forest model is better suited for predicting heart disease.

