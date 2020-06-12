#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
#from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os
import pickle


# In[8]:


cars = pd.read_csv("/Users/vamshi/Desktop/modifiedcars_deploy.csv",encoding="latin-1")


# In[9]:


cars.head(5)


# In[10]:


cars = cars.drop('name',1)
cars = cars.drop('Unnamed: 0',1)
cars = cars.drop('model',1)
cars = cars.drop('brand',1)




# In[11]:


cars.head(10)


# In[12]:


cars['vehicleType'].value_counts()


# In[48]:


cars.isnull().sum()


# In[13]:


cars = cars[['vehicleType','yearOfRegistration','gearbox','powerPS','kilometer','monthOfRegistration','fuelType','notRepairedDamage','price']]


# In[14]:


cars1 = preprocessing.LabelEncoder()
cars["vehicleType"] = cars1.fit_transform(cars["vehicleType"])


# In[32]:


cars["monthOfRegistration"].value_counts()


# In[80]:


cars["brand1"].value_counts()


# In[16]:


cars["gearbox"] = cars1.fit_transform(cars["gearbox"])
cars["fuelType"] = cars1.fit_transform(cars["fuelType"])
cars["notRepairedDamage"] = cars1.fit_transform(cars["notRepairedDamage"])

cars1.head(5)
# In[17]:


cars.head(10)


# In[18]:


X = cars.iloc[:,:-1].values
y = cars.iloc[:,8].values


# In[35]:


cars['vehicleType']


# In[22]:


X_train


# In[23]:


regressor = RandomForestRegressor(criterion = 'mse', max_depth = 10,n_estimators = 200,min_samples_leaf=3,min_samples_split=3)
model = regressor.fit(X_train,y_train)


# In[24]:


y_predict = model.predict(X_test)
y_predict 


# In[25]:


y_predict,y_test


# In[26]:


from sklearn.metrics import r2_score,mean_squared_error
print (f' Accuracy of the model : {r2_score(y_predict,y_test)}')
print (f' MSE of the model : {mean_squared_error(y_predict,y_test)}')


# In[104]:


pip install Flask


# In[27]:


pickle.dump(model,open('car_model2.pkl','wb'))


# In[28]:


model = pickle.load(open('car_model2.pkl','rb'))


# In[30]:


print(model.predict([[6, 2010, 1,100,90000,0,0,0]]))


# In[ ]:





# In[18]:





# In[20]:





# In[ ]:




