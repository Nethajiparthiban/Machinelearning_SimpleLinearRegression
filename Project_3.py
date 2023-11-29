#Importing the modules..
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Reading the data set..
data=pd.read_csv("D:\Git\Git-Projects\car_stopping.csv")
#print(data.head(5))
#Info about the data set
#print(data.info())
#describing the data set.
#print(data.describe())
#Checking the Null values
#print(data.isnull().sum())
#Assiging the values for X and Y
X=data['Speed']
Y=data['Distance']
#print(X)
#print(Y)
X=data.iloc[:,:-1]
Y=data.iloc[:,-1]
#print(X)
#print(Y)
# Seprating data for traing and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
#print(x_train)
#print(x_test)
#print(y_test)
#print(y_train)
#Fitting the dataset to linear model algoritham
from sklearn.linear_model import LinearRegression
stud=LinearRegression()
stud.fit(x_train,y_train)
#Predicting the data
y_pred=stud.predict(x_test)
#Checking the Accuracy for Linear model
from sklearn.metrics import mean_squared_error,r2_score
rscore=r2_score(y_test,y_pred)
scr=rscore*100
print(scr.round(),'%')

