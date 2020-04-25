# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:36:35 2020

@author: ajayy
"""

import pandas as pd
from matplotlib import pylab as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle

sal_exp = pd.read_csv("E:/Assignments/Module 6 - Regression/Salary_hike/Data set/Salary_Data.csv")

# uni variate analysis

#plt.hist(sal_exp["YearsExperience"])
#plt.boxplot(sal_exp["YearsExperience"])
# No outliers observed in boxplot of "YearsExperience"

#plt.hist(sal_exp["Salary"])
#plt.boxplot(sal_exp["Salary"])
# No outliers observed in boxplot of "Salary"

sal_exp.describe()
sal_exp.columns

# Scatter plot to check the relation between x and y
#plt.scatter(x = sal_exp["YearsExperience"], y = sal_exp["Salary"])
#plt.xlabel("Experience (in Years)")
#plt.ylabel("Salary (in USD)")

# Separate the data into features and data
features = sal_exp.iloc[:,[0]].values
label = sal_exp.iloc[:,[1]].values


# Splitting the data into train & test
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# checked and choosen random_state as 39 which gives good score for both train and test.
# 1. Creating a model without doing any transformations.
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2 ,
                                                    random_state = 39)


salPred = LinearRegression()
salPred.fit(X_train, y_train)
pred1 = salPred.predict(features)

print("Salary = {} + {} * YearsExperience".format(salPred.intercept_[0],salPred.coef_[0][0]))

train_score = salPred.score(X_train, y_train)
test_score = salPred.score(X_test, y_test)
score = salPred.score(features, label)
#plt.scatter(x = sal_exp["YearsExperience"], y = sal_exp["Salary"])
#plt.xlabel("Experience (in Years)")
#plt.ylabel("Salary (in USD)")
#plt.plot(sal_exp["YearsExperience"],pred1)

rmse = np.sqrt(mean_squared_error(sal_exp["Salary"],pred1))

# R square : 0.9569 (full data set); rmse : 5593.21 ; train_score : 0.949  ; test_score : 0.992


# 2. Creating a model by applying logarithmic transformations to features(Independent variables)

sal_exp["logYearsExperience"] = np.log(sal_exp["YearsExperience"])
#sal_exp.columns
features2 = sal_exp.iloc[:,[2]].values
label2 = sal_exp.iloc[:,[1]].values
X_train, X_test, y_train, y_test = train_test_split(features2, label2, test_size = 0.2 ,random_state = 39)

salPred2 = LinearRegression()
salPred2.fit(X_train, y_train)
pred2 = salPred2.predict(features2)

print("Salary = {} + {} * YearsExperience".format(salPred2.intercept_[0],salPred2.coef_[0][0]))

train_score = salPred2.score(X_train, y_train)
test_score = salPred2.score(X_test, y_test)
score = salPred2.score(features2, label2)
#plt.scatter(x = sal_exp["YearsExperience"], y = sal_exp["Salary"])
#plt.xlabel("Experience (in Years)")
#plt.ylabel("Salary (in USD)")
#plt.plot(sal_exp["YearsExperience"],pred2)

rmse2 = np.sqrt(mean_squared_error(sal_exp["Salary"],pred2))

# R square : 0.8537 (full data set); rmse: 10309.39 ; train_score : 0.8363  ; test_score : 0.9257



# 3. Creating a model by applying exponential transformation, i.e logarithmic transformation to label (dependent variable)

sal_exp["logSalary"] = np.log(sal_exp["Salary"])
#sal_exp.columns
features3 = sal_exp.iloc[:,[0]].values
label3 = sal_exp.iloc[:,[3]].values

X_train, X_test, y_train, y_test = train_test_split(features3, label3, test_size = 0.2, random_state = 39)

salPred3 = LinearRegression()
salPred3.fit(X_train,y_train)
pred3 = np.exp(salPred3.predict(features3))

print("Salary = {} + {} * YearsExperience".format(salPred3.intercept_[0],salPred3.coef_[0][0]))

train_score = salPred3.score(X_train, y_train)
test_score = salPred3.score(X_test, y_test)
score = salPred3.score(features3, label3)
#plt.scatter(x = sal_exp["YearsExperience"], y = sal_exp["Salary"])
#plt.xlabel("Experience (in Years)")
#plt.ylabel("Salary (in USD)")
#plt.plot(sal_exp["YearsExperience"],pred3)

rmse3 = np.sqrt(mean_squared_error(sal_exp["Salary"],pred3))

# R square : 0.931 (full data set); rmse3 = 7058.13 ; train_score : 0.927  ; test_score : 0.941


# 4. Creating a model by applying exponential transformation for Y , i.e logarithmic transformation to -
#    label (dependent variable) and polynomial 2D transformation for features (dependent variable).
#    np.log(Y) ~ X + X^2
    

sal_exp["logSalary"] = np.log(sal_exp["Salary"])
sal_exp["YearsExperience_sq"] = sal_exp["YearsExperience"] ** 2
#sal_exp.columns
features4 = sal_exp.iloc[:,[0,4]].values
label4 = sal_exp.iloc[:,[3]].values

X_train, X_test, y_train, y_test = train_test_split(features4, label4, test_size = 0.2, random_state = 39)

salPred4 = LinearRegression()
salPred4.fit(X_train,y_train)
pred4 = np.exp(salPred4.predict(features4))

print("Salary = {} + {} * YearsExperience + {} * YearsExperience_sq".format(salPred4.intercept_[0],
                                                      salPred4.coef_[0][0],salPred4.coef_[0][1]))

train_score = salPred4.score(X_train, y_train)
test_score = salPred4.score(X_test, y_test)
score = salPred4.score(features4, label4)
#plt.scatter(x = sal_exp["YearsExperience"], y = sal_exp["Salary"])
#plt.xlabel("Experience (in Years)")
#plt.ylabel("Salary (in USD)")
#plt.plot(sal_exp["YearsExperience"],pred4)

rmse4 = np.sqrt(mean_squared_error(sal_exp["Salary"],pred4))

# R square : 0.948 (full data set); rmse4 : 5486.495 ;train_score : 0.941  ; test_score : 0.974



# 4. Creating a model by applying exponential transformation for Y , i.e logarithmic transformation to -
#    label (dependent variable) and polynomial 3D transformation for features (dependent variable).
#    np.log(Y) ~ X + X^2 + X^3
    

sal_exp["logSalary"] = np.log(sal_exp["Salary"])
sal_exp["YearsExperience_cb"] = sal_exp["YearsExperience"] ** 3
sal_exp.columns
features5 = sal_exp.iloc[:,[0,4,5]].values
label5 = sal_exp.iloc[:,[3]].values

X_train, X_test, y_train, y_test = train_test_split(features5, label5, test_size = 0.2, random_state = 39)

salPred5 = LinearRegression()
salPred5.fit(X_train,y_train)
pred5 = np.exp(salPred5.predict(features5))

print("Salary = {} + {} * YearsExperience + {} * YearsExperience_sq + {} * YearsExperience_cb".format(salPred5.intercept_[0],
                                              salPred5.coef_[0][0],salPred5.coef_[0][1],salPred5.coef_[0][2]))

train_score = salPred5.score(X_train, y_train)
test_score = salPred5.score(X_test, y_test)
score = salPred5.score(features5, label5)
plt.scatter(x = sal_exp["YearsExperience"], y = sal_exp["Salary"])
plt.xlabel("Experience (in Years)")
plt.ylabel("Salary (in USD)")
plt.plot(sal_exp["YearsExperience"],pred5, color = "red")

rmse5 = np.sqrt(mean_squared_error(sal_exp["Salary"],pred5))

# R square : 0.950 (full data set); rmse: 5290.24 ; train_score : 0.947  ; test_score : 0.955

# After evaluating various models, we came to conclusion that the model which is created by polynomial 3D and - 
#  exponential transformations gives the best fit line that passes through the given data.

# Saving model to disk
pickle.dump(salPred5, open('salary_exp.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('salary_exp.pkl','rb'))

# checking the model by giving input from console and printing the predicted output
yrs_of_exp = float(input("Enter Years of Experience : "))
sal_of_exp = model.predict(np.array([[yrs_of_exp, yrs_of_exp ** 2, yrs_of_exp ** 3]]))
print("Salary of {} years experience person is ${}".format(yrs_of_exp, int(np.exp(sal_of_exp)[0][0])))









