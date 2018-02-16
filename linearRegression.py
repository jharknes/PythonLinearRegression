# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:04:45 2018

@author: joshu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Salary_Data.csv')
#matrix of independent variables
x = dataset.iloc[:,:-1].values #first side of comma is rows, second is columns
#now dependent 
y = dataset.iloc[:,1].values

#splitting the data set into a training  set and validation set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

#fit the simple linear regression to the training set###############################################
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predict test results
y_predictions = regressor.predict(x_test)

#visualization of training set data
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color='blue')#makes our training results test
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.show()
#plt.savefig('PythonsSLRjpeg.png')

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='orange')#makes our training results test
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.show()
plt.savefig('PythonsSLR.png')