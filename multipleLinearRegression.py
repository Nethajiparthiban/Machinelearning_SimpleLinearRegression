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