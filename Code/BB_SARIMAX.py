#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SARIMAX Model of Bed and Bath category 


# In[1]:


import pandas as pd
import numpy as np
import datetime
import plotly.express as px


# In[5]:


bed_bath = pd.read_csv('bed_bath_table_data.csv')

bb_data = bed_bath
# define short column names
bb_data.columns = ['date', 'category', 'sales', 'revenue']
# Changing date data type to datetime
bb_data['date'] = pd.to_datetime(bb_data['date'], format="%Y-%m-%d")

bb_categ = bb_data[['date','sales']]

# make date as index
bb_categ = bb_categ.set_index('date')
bb_categ.head()


# In[6]:


# plot the sales
px.line(bb_categ)


# In[8]:


# Seasonal decomposition of sales to find trend and seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
dec = seasonal_decompose(bb_categ['sales'], model='additive', period=4)
print(dec.plot())


# In[24]:


# Split data into train and test
train = bb_categ[:474]
test = bb_categ[474:]
print(train.shape)
print(test.shape)
test.head()


# In[27]:


# Model building using train set
import statsmodels.api as sm

model = sm.tsa.SARIMAX(train['sales'], order= (1,1,1),
                       seasonal_order=(1,0,2,12))
modelFit = model.fit() # Fit the model using standard params

print(modelFit.summary())


# In[17]:


# Residuals
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
rsd = modelFit.resid
ax = rsd.plot.kde()
ax


# In[18]:


# Predictions using test
fcst = modelFit.forecast(steps=len(test))
fcst 


# In[19]:


# Computing MSE
from sklearn.metrics import mean_squared_error 
MSE = mean_squared_error(test['sales'], fcst)
MSE


# In[26]:


#plotting actual test set(blue line) vs predicted(red line)
px.line(test, y=['sales', fcst])


# In[ ]:




