import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df=pd.DataFrame({
    'area':[2600,3000,3200,3600,4000],
    'price':[550000,565000,610000,680000,725000]
})
#print(df)
x=df['area']
y=df['price']
plt.xlabel('area')
plt.ylabel('price')
plt.title('Housing price')
plt.plot(x,y,color='green',marker='+')
plt.show()
reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
reg.predict([[3300]])#628715.75342466
#as we the formula y=mx+b
#here m=slope and b is intercept m is nothing but coef_ and b is intercept
reg.coef_#135.78767123
reg.intercept_#180616.43835616432
#lets cross verify our prediction
#print(135.78767123*3300+180616.43835616432)#628715.7534151643
#lets create another set of values df
df1=pd.DataFrame({
    'area':[1000,1500,2300,3540,4120,4560,5490,3460,4750,2300,9000,8600,7100]
})
x=reg.predict(df1)
df1['price']=x
#lets plot the scatter and line plot after predicting the values
plt.xlabel('Area')
plt.ylabel('Price')
x=df['area']
y=reg.predict(df[['area']])
plt.title('Housing price prediction')
plt.scatter(x=df['area'],y=df['price'],color='orange',marker='+')
plt.plot(x,y,color='blue')
plt.show()

