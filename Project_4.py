#Importing the modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Reading the data set
data=pd.read_csv("D:\Git\Git-Projects\Deaths.csv")
#print(data.head(5))
#print(data.info())
#print(data.describe())
#print(data.isnull().sum())
#Seggrigating the data and assigning x and y
X=data.iloc[:,:-1]
Y=data.iloc[:,-1]
#print(Y)
#print(X)
#Assining for training
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
#print(x_test)
#print(y_test)
#print(y_train)
#print(x_train)
#Fitting to the alogoritham module.
from sklearn.linear_model import LinearRegression
stud=LinearRegression()
stud.fit(x_train,y_train)
#Prdicting the data.
y_pred=stud.predict(x_test)
#Checking the accuracy.
from sklearn.metrics import mean_squared_error,r2_score
rscore=r2_score(y_test,y_pred)
rs=rscore*100
print(rs.round(),'%')

