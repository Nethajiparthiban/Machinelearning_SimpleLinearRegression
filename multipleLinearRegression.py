#importing required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Reading the dataset.
data=pd.read_csv("D:\Git\Git-Projects\ML--Simple-Linear-Regression\Startups.csv")
#print(data.head(5))
#info about dataset
#print(data.info())
#describing about data set.
#print(data.describe())
#Checking the Null values
#print(data.isnull().sum())
#seggrigating the values and assigining values to X and Y.
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values
#Encoding the string values..
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(X))
#print(x)
#splitting the data set.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=1)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)
#Fitting the dataset for training
from sklearn.linear_model import LinearRegression
stud=LinearRegression()
stud.fit(x_train,y_train)
#Predicting the data set.
y_pred=stud.predict(x_test)
#print(y_pred)
#print(stud.score(x_test,y_test))
#Plotting the data
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
#print(plt.show())