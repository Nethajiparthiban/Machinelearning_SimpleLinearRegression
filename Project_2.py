#importing the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Reading the Data set
data=pd.read_csv("D:\Git\Git-Projects\ML--Simple-Linear-Regression\salary_data.csv")
#checking the imported data
#print(data.head(3))
#checking shape of  the data.
#print(data.shape)
#checking the Info of the data.
#print(data.info())
#describing the data set.
#print(data.describe())
#Checking the Null values
#print(data.isnull().sum())
#Seggrigating the data set for x and y.
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values
#print(Y)
#print(X)
#Model selection importing.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)
#Training the Data set[choosing the Algoritham model]
from sklearn.linear_model import LinearRegression
stud=LinearRegression()
stud.fit(x_train,y_train)
#Predicting all the data
y_predict=stud.predict(x_test)
#print(y_predict)
#Checking the Accuracy
from sklearn.metrics import mean_squared_error,r2_score
rscore=r2_score(y_test,y_predict)
r=rscore*100
print(r.round(),'%')

#Conclussion:-
  '''For the salary data Simple Linear regression model is very much suitable'''

