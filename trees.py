#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from statistics import mean
from sklearn import linear_model

data = pd.read_csv('2019.csv')

features = data[['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices','Generosity', 'Perceptions of corruption']]
happiness = data['Score']

decision_tree = tree.DecisionTreeRegressor(random_state=0)
decision_tree.fit(features, happiness)
hypothesis = decision_tree.predict(features)
mean_squared_error_dt = mean_squared_error(hypothesis, happiness)
print('mean squared error of decision tree: ', mean_squared_error_dt)


random_forest = RandomForestRegressor(n_estimators=100, random_state=0)
random_forest.fit(features, happiness)
hypothesis = random_forest.predict(features)
mean_squared_error_rf = mean_squared_error(hypothesis, happiness)
print('mean squared error of random forest: ', mean_squared_error_rf)

kfold = KFold(n_splits=5)

def kfold_for_decision_tree():
    mean_squarred = []
    for train_indices, test_indices in kfold.split(features, happiness):
        decision_tree = tree.DecisionTreeRegressor(random_state = 0)
        decision_tree.fit(features.loc[train_indices], happiness.loc[train_indices])
        predictions = decision_tree.predict(features.loc[test_indices])
        mse = mean_squared_error(predictions, happiness.loc[test_indices])
        mean_squarred.append(mse)

    print('Decision Tree KFold Average', round(mean(mean_squarred), 2))

kfold_for_decision_tree()
        
def kfold_for_random_forest():
    mean_squarred = []
    for train_indices, test_indices in kfold.split(features, happiness):
        random_forest = RandomForestRegressor(n_estimators=100, random_state=0)
        random_forest.fit(features.loc[train_indices], happiness.loc[train_indices])
        predictions = random_forest.predict(features.loc[test_indices])
        mse = mean_squared_error(predictions, happiness.loc[test_indices])
        mean_squarred.append(mse)

    print('Random Forest KFold Average', round(mean(mean_squarred), 2))


kfold_for_random_forest()

def kfold_for_mutiple_linear_regression():
    mean_squarred = []
    for train_indices, test_indices in kfold.split(features, happiness):
        model = linear_model.LinearRegression()
        model.fit(features.loc[train_indices], happiness.loc[train_indices])
        predictions = model.predict(features.loc[test_indices])
        mse = mean_squared_error(predictions, happiness.loc[test_indices])
        mean_squarred.append(mse)

    print('Mutiple Linear Regression KFold Average', round(mean(mean_squarred), 2))

kfold_for_mutiple_linear_regression()

def cross_val_for_random_forest():
    random_forest =random_forest = RandomForestRegressor(n_estimators=100, random_state=0)
    random_forest_prediction_score = cross_validate(random_forest, features,happiness, cv = 5,scoring=('r2', 'neg_mean_squared_error')) 
    print(random_forest_prediction_score)

cross_val_for_random_forest()

def cross_val_for_decision_tree():
   decision_tree = tree.DecisionTreeRegressor(random_state=0)
   decision_tree_prediction_score = cross_validate(decision_tree, features,happiness, cv = 5,scoring=('r2', 'neg_mean_squared_error'))
   print(decision_tree_prediction_score)

cross_val_for_decision_tree()









