#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Fastest Growing Category: Construction and Tools


# In[8]:


import pandas as pd
import numpy as np
import datetime
import plotly.express as px


# In[7]:


# load data
construction_dat = pd.read_csv("construction_tools_lights_data.csv")

construction_dat.columns = ['date', 'category', 'sales', 'revenue']
# Changing date data type to datetime
construction_dat['date'] = pd.to_datetime(construction_dat['date'], format="%Y-%m-%d")

const = construction_dat[['date','sales']]

# make date as index
const = const.set_index('date')
const.head()


# In[9]:


# Plot the time series
px.line(const)


# In[10]:


# Seasonal decomposition of sales to find trend and seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
dec = seasonal_decompose(const, model='additive', period=4)
print(dec.plot())

# seasonal data and trend visible


# In[55]:


# Split data into train and test
train = const[:55]
test = const[55:]
print(train.shape)
print(test.shape)
test.head()


# In[56]:


# Model building using train set
import statsmodels.api as sm

model = sm.tsa.SARIMAX(train, order= (0,1,1),
                       seasonal_order=(0,0,1,12))
 
modelFit = model.fit() # Fit the model using standard params

print(modelFit.summary())


# In[57]:


# Residuals
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
rsd = modelFit.resid
ax = rsd.plot.kde()
ax


# In[58]:


# Predictions using test
fcst = modelFit.forecast(steps=len(test))
fcst 


# In[59]:


# Computing MSE
from sklearn.metrics import mean_squared_error 
MSE = mean_squared_error(test, fcst)
MSE


# In[60]:


#plotting actual test set(blue line) vs predicted(red line)
px.line(test, y=['sales', fcst])


# In[ ]:


# USING WHOLE DATASET TO BUILD MODEL


# In[61]:


# Model building using train set
import statsmodels.api as sm

model = sm.tsa.SARIMAX(const, order= (0,1,1),
                       seasonal_order=(0,0,1,12))
 
modelFit = model.fit() # Fit the model using standard params

print(modelFit.summary())


# In[62]:


# Residuals
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
rsd = modelFit.resid
ax = rsd.plot.kde()
ax


# In[63]:


# Predictions for next three months
fcst = modelFit.forecast(steps=90)
fcst 

