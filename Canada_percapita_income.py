import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv(r"C:\Users\Netha\Ananconda_onlineclass\Mission learning\ML from codebasics\Machine Learning\Linear_regression\canada_per_capita_income.csv")
df['percapita income']=df['per capita income (US$)']
df.drop('per capita income (US$)',inplace=True,axis=1)
#print(df)
plt.xlabel('year')
plt.ylabel('Percapita')
plt.title("ploting b4 data prediction")
plt.scatter(x=df['year'],y=df['percapita income'],marker='o',color='red')
plt.plot(df['year'],df['percapita income'])
plt.show()
#fitting to algoritham
reg=linear_model.LinearRegression()
reg.fit(df[['year']],df['percapita income'])
reg.predict([[2020]])#41288.69409442
reg.coef_#it means slope#828.46507522
reg.intercept_#-1632210.7578554575
#as we know the formula is y=m*x+b
#print(828.46507522*2020-1632210.7578554575)#41288.694088942604
#now it is time to preditciting the new values
new_df=pd.DataFrame({
    "year":[2017,2018,2019,2020,2021,2022,2023,2024,2025]
})
x=reg.predict(new_df)
new_df['percapita income']=x
#ploting chart after prediction
new_df1=pd.concat([df,new_df],ignore_index=True)

plt.xlabel('year')
plt.ylabel('percapita income')
plt.title('After prediction')
plt.scatter(x=new_df1['year'],y=new_df1['percapita income'],color='blue',marker='+')
plt.plot(new_df1['year'],reg.predict(new_df1[['year']]))
plt.show()
import pickle

with open('pick_pickle','wb') as k:
    pickle.dump(reg,k)
with open('pick_pickle','rb') as k:
    pred=pickle.load(k)

print(pred.predict([[2025]]))