# importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df=pd.read_csv(r"C:\Users\Netha\Ananconda onlinclass\Mission learning\ML from codebasics\canada_per_capita_income.csv")
#print(df.head())
plt.title('Canada percapita Income')
plt.xlabel('Year')
plt.ylabel('Income')
plt.scatter(df.year,df.percapit,color='red',marker="o")
plt.show()
#Y=mx+b
reg=linear_model.LinearRegression()
reg.fit(df[['year']],df.percapit)
print(reg.predict(np.array(2022).reshape(-1,1)))
print(reg.coef_)
print(reg.intercept_)

plt.scatter(df.year,df.percapit,color='red',marker='o')
plt.plot(df.year,reg.predict(df[['year']]),color='blue')
plt.show()

