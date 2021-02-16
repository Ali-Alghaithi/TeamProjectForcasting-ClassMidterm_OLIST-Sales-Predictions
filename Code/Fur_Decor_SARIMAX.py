#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# OLS for Furniture and Decor


# In[2]:


import pandas as pd
import numpy as np
import datetime
import plotly.express as px


# In[3]:


decor_data = pd.read_csv('furniture_decor.csv')

decor = decor_data
# define short column names
decor.columns = ['date', 'category', 'sales', 'revenue']
# Changing date data type to datetime
decor['date'] = pd.to_datetime(decor['date'], format="%Y-%m-%d")

decor_dat = decor[['date','sales']]

# make date as index
decor_dat = decor_dat.set_index('date')


# In[27]:


decor_dat.head()


# In[4]:


px.line(decor_dat)


# In[55]:


# Seasonal decomposition of sales to find trend and seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
dec = seasonal_decompose(decor['sales'], model='additive', period=4)
print(dec.plot())


# In[15]:


# Split data into train and test
train = decor[:479]
test = decor[479:]
print(train.shape)
print(test.shape)


# In[16]:


import statsmodels.api as sm

model = sm.tsa.SARIMAX(train['sales'], order= (1,1,1),
                       seasonal_order=(1,0,2,12))
 # specifying an ARIMA(1,1,0) model
modelFit = model.fit() # Fit the model using standard params
#res = modelFit.resid   # store the residuals as res
print(modelFit.summary())


# In[24]:


# Residuals
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
rsd = modelFit.resid
ax = rsd.plot.kde()
ax


# In[ ]:





# In[23]:


# Predictions using test
fcst = modelFit.forecast(steps=len(test))
fcst 


# In[27]:


from sklearn.metrics import mean_squared_error 
MSE = mean_squared_error(test['sales'], fcst)
MSE


# In[26]:


#plotting actual test set(blue line) vs predicted(red line)
px.line(test, x='date', y=['sales', fcst])


# In[ ]:




