#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_validate
data = pd.read_csv('2019.csv')


features = data[['GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices','Generosity', 'Perceptions of corruption']]
happiness = data[['Score']]


model = linear_model.LinearRegression()

model.fit(features, happiness)
    
thetas = model.coef_
theta_0 = model.intercept_


def predictions():
    predictions_by_model = np.array([])
    #Reads each row of file as an array
    for data in np.array(features):
        to_append = np.array(model.predict(np.array(data).reshape(1,-1)))
        predictions_by_model = np.append(predictions_by_model, to_append)
    return predictions_by_model
    
        
        

      
predictions_from_model = predictions()

def square_the_sum(sum):
    return (sum**2)   
    
def cost_function():
    sum = 0
    happiness_array = np.array(happiness)
    number_of_countries = 156
    for index in range(number_of_countries):
        sum+= predictions_from_model[index] - happiness_array[index]
    squared_sum = square_the_sum(sum)
    return ((1.0/(2*number_of_countries)))*squared_sum
    
cost = cost_function()
print(cost)                          
print("thetas: =", thetas)
print("theta_0 =", theta_0)

k_model = linear_model.LinearRegression()
k_predictions_score = cross_validate(k_model, features,happiness, cv = 5,scoring=('r2', 'neg_mean_squared_error'))

print(k_predictions_score)


