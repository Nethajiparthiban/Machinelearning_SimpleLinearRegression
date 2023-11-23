import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv("D:\Git\Git-Projects\Test.csv")
data.shape
#Seggrigating data set to (X--input/Independent variable) and (Y--input/dependent variable)
X=data.iloc[:,:-1]
Y=data.iloc[:,-1]
#Splitting the Data set for train and testing.
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
#training the Data (Choosing the Algoritham Model)
stud=LinearRegression()
stud.fit(x_train,y_train)
y_predict=stud.predict(x_test)
#Checking the Accuracy
rscore=r2_score(y_test,y_predict)
r=rscore*100
print(r.round(),'%')