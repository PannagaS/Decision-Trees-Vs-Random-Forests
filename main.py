import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

data = pd.read_csv('heart.csv')
category_variables = ['Sex',
'ChestPainType',
'RestingECG',
'ExerciseAngina',
'ST_Slope'
]

data = pd.get_dummies(data = data,
                         prefix = category_variables,
                         columns = category_variables)

features = [i for i in data.columns if i not in 'HeartDisease']

#splitting data into train and test set (85% training data, 15% testing data)
RANDOM_STATE = 55  #for reproducability
X_train, X_val, y_train, y_val = train_test_split(data[features], data['HeartDisease'], train_size = 0.80, random_state = RANDOM_STATE)
print('X train shape ', X_train.shape)
print('X test shape ',X_val.shape)
print('y train shape ',y_train.shape)
print('y test shape ',y_val.shape)

#building decision tree
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]
max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None]
train_accuracy_for_minsample =[]
val_accuracy_for_minsample=[]
print("building optimal dt based on min_sample_split")
for min_samples_split in min_samples_split_list :
    model = DecisionTreeClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE)
    model.fit(X_train, y_train)
    heart_predictions_train = model.predict(X_train)
    heart_predictions_validation = model.predict(X_val)
    train_accuracy_for_minsample.append(accuracy_score(heart_predictions_train, y_train))
    val_accuracy_for_minsample.append(accuracy_score(heart_predictions_validation,y_val))



#plotting the data for visualization
plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
# plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
# plt.plot(accuracy_train)
# plt.plot(accuracy_validation)
plt.plot( min_samples_split_list,train_accuracy_for_minsample, lw=1,color='red')
plt.plot(min_samples_split_list,val_accuracy_for_minsample, lw=1,color='blue')
plt.legend(['Train','Validation'])
plt.grid(True)
plt.show()

print("building optimal dt based on max_depth_list")

train_accuracy_for_maxdepth =[]
val_accuracy_for_maxdepth = []
for max_depth in max_depth_list :
    model = DecisionTreeClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE)
    model.fit(X_train, y_train)
    heart_predictions_train = model.predict(X_train)
    heart_predictions_validation = model.predict(X_val)
    train_accuracy_for_maxdepth.append(accuracy_score(heart_predictions_train, y_train))
    val_accuracy_for_maxdepth.append(accuracy_score(heart_predictions_validation,y_val))

#plotting the data for visualization
plt.title('Train x Validation metrics')
plt.xlabel('max_depth_samples')
plt.ylabel('accuracy')
plt.plot( max_depth_list,train_accuracy_for_maxdepth, lw=1,color='red')
plt.plot(max_depth_list,val_accuracy_for_maxdepth, lw=1,color='blue')
plt.legend(['Train','Validation'])
plt.grid(True)
plt.show()

"""
from the above graphs
optimal value for min_samples_split = 50 
optimal value for max_depth = 3

"""

decision_tree_model = DecisionTreeClassifier(min_samples_split = 50,
                                             max_depth = 3,
                                             random_state = RANDOM_STATE).fit(X_train,y_train)
print(f"Metrics for single decision tree train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}")
print(f"Metrics for single decision tree validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_val),y_val):.4f}")


"""
Random forest implementation
"""

n_estimators_list = [10,50,100,500]
#deciding min number of splits
accuracy_train =[]
accuracy_val = []
for min_samples in min_samples_split_list :
    model = RandomForestClassifier(min_samples_split = min_samples,random_state = RANDOM_STATE)
    model.fit(X_train, y_train)
    prediction_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train.append(accuracy_score(prediction_train, y_train))
    accuracy_val.append(accuracy_score(predictions_val,y_val))

plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_train, lw=0.7,color='green')
plt.plot(accuracy_val,lw=0.7,color='blue')
plt.legend(['Train','Validation'])
plt.grid(True)
plt.show()

#deciding max depth
accuracy_list_train = []
accuracy_list_val = []

for max_depth in max_depth_list:
    model = RandomForestClassifier(max_depth = max_depth,random_state = RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_train)
plt.plot(accuracy_val)
plt.legend(['Train','Validation'])

plt.show()



#checking for optimal n_estimator value
accuracy_list_train = []
accuracy_list_val = []
for n_estimators in n_estimators_list:
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   random_state = RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
plt.plot(accuracy_list_train,color='orange')
plt.plot(accuracy_list_val,color='blue')
plt.legend(['Train','Validation'])
plt.grid(True)
plt.show()


random_forest_model = RandomForestClassifier(n_estimators = 50,
                                             max_depth = 32,
                                             min_samples_split = 8, random_state=RANDOM_STATE).fit(X_train,y_train)
print(f"Metrics for random forest train:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_train),y_train):.4f}\nMetrics for random forest test:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_val),y_val):.4f}")

"""
using gridsearch 


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

param_grid = {'min_samples_split' : [2,10, 30, 50, 100, 200, 300, 700],'max_depth':[1,2, 3, 4, 8, 16, 32, 64, None]}
dt = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, refit='accuracy')
grid_search.fit(X_train, y_train)
best_estimator = grid_search.best_estimator_
best_score = grid_search.best_score_
y_pred = best_estimator.predict(X_val)
accuracy = best_estimator.score(X_val, y_val)
print(f"The best estimator is {best_estimator} with an accuracy of {accuracy:.2f}")
"""