#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
df=pd.read_csv("Dataset .csv")
df=pd.DataFrame(df)
lc=LabelEncoder()
df['City']=lc.fit_transform(df['City'])
df['Restaurant Name']=lc.fit_transform(df['Restaurant Name'])
df['Address']=lc.fit_transform(df['Address'])
df['Cuisines']=lc.fit_transform(df['Cuisines'])
df['Average Cost for two']=lc.fit_transform(df['Average Cost for two'])
df['Locality']=lc.fit_transform(df['Locality'])
df['Currency']=lc.fit_transform(df['Currency'])
df['Has Table booking']=lc.fit_transform(df['Has Table booking'])
df['Has Online delivery']=lc.fit_transform(df['Has Online delivery'])
df['Is delivering now']=lc.fit_transform(df['Is delivering now'])
df['Switch to order menu']=lc.fit_transform(df['Switch to order menu'])
df['Rating color']=lc.fit_transform(df['Rating color'])
df['Rating text']=lc.fit_transform(df['Rating text'])
x=df.iloc[:,[1,2,3,4,5,9,10,11,12,13,14,15,16,17,19,20]]
y=df.iloc[0:,18]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model1 = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5,random_state=42)
model1.fit(x_train,y_train)
ypred=model1.predict(x_test)
r2=r2_score(y_test,ypred)
mae=mean_absolute_error(y_test,ypred)
mse=mean_squared_error(y_test,ypred)
print("ThePerformance of the Decision Tree Regressor on the aggregate rating:")
print("R2 Score",r2)
print("Mean absolute Error:",mae)
print("Mean Squared Error:",mse)


# In[10]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
r2_scores = cross_val_score(model1, x, y, cv=kf, scoring='r2')
mae_scores = cross_val_score(model1, x, y, cv=kf, scoring=make_scorer(mean_absolute_error))
mse_scores = cross_val_score(model1, x, y, cv=kf, scoring=make_scorer(mean_squared_error))
print(r2_scores,mae_scores,mse_scores)


# In[11]:


rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

# Train the model
rf_regressor.fit(x_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(x_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")


# In[ ]:




