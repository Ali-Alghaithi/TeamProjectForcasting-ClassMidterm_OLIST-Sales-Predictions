#!/usr/bin/env python
# coding: utf-8

# In[1]:


# OLS Furniture and Decor
import pandas as pd
import numpy as np
import datetime
import plotly.express as px


# In[2]:


decor_data = pd.read_csv('furniture_decor.csv')

decor = decor_data
# define short column names
decor.columns = ['date', 'category', 'sales', 'revenue']
# Changing date data type to datetime
decor['date'] = pd.to_datetime(decor['date'], format="%Y-%m-%d")
#add the year, month and day of the week
decor['year'] = decor['date'].dt.year
decor['month'] = decor['date'].dt.month
decor['dayofweek'] = decor['date'].dt.weekday

# add weekends and black friday
decor = decor.assign(weekend = lambda dataframe:
                dataframe['dayofweek'].map(lambda dayofweek:
                                           "Yes" if ((dayofweek == 5) | (dayofweek == 6))
                                           else "No"))

decor = decor.assign(black_friday = lambda dataframe:
                dataframe['sales'].map(lambda sales:
                                           "Yes" if (sales == max(decor['sales']))
                                           else "No"))

decor.head()


# In[5]:


# FURNITURE AND DECOR


# In[3]:


px.line(decor, x='date', y= 'sales')


# In[24]:


# Seasonal decomposition of sales to find trend and seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
dec = seasonal_decompose(decor['sales'], model='additive', period=4)
print(dec.plot())


# In[19]:


# Split data into train and test
train = decor[:479]
test = decor[479:]
print(train.shape)
print(test.shape)


# In[17]:


decor.tail()


# In[20]:


# building the model
import patsy as pt
y, x = pt.dmatrices("sales ~ C(year) + C(month)+ C(dayofweek) + weekend + black_friday",
                   data = train)

import statsmodels.api as sm
# Declare the model, and create an instance of the OLS
#  class
model = sm.OLS(endog = y, exog = x)
# Fit the model, optionally using specific parameters
modelFit = model.fit()
modelFit.summary()


# In[21]:


# Residuals
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
res = modelFit.resid
res = pd.Series(res)
ax = res.plot.kde()
ax # The errors are normally distributed with mean 0


# In[22]:


# Prediction using test data
test_x = pt.build_design_matrices([x.design_info], test)
pred = modelFit.predict(test_x).squeeze()

# plot test actual and predicted
px.line(test, x='date', y=['sales', pred])


# In[23]:


# MSE
from sklearn.metrics import mean_squared_error 
MSE = mean_squared_error(test['sales'], pred)
MSE


# In[49]:


# USING THE WHOLE DATASET TO BUILD THE MODEL AND MAKE THE  MONTHS PREDICTIONS


# In[24]:


# building the model
import patsy as pt
y, t = pt.dmatrices("sales ~ C(year) + C(month)+ C(dayofweek) + weekend + black_friday",
                   data = decor)

import statsmodels.api as sm
# Declare the model, and create an instance of the OLS
#  class
Model = sm.OLS(endog = y, exog = t)
# Fit the model, optionally using specific parameters
ModelFit = Model.fit()
ModelFit.summary()


# In[25]:


# Residuals
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
resid = ModelFit.resid
resid = pd.Series(resid)
ax = resid.plot.kde()
ax


# In[26]:


# Prediction using rthe whole dataset
Pred = ModelFit.predict(t).squeeze()

# plot  actual and predicted
px.line(decor, x='date', y=['sales', Pred])


# In[27]:


# Make input data for predictions
# make a list of 90 days
from datetime import datetime
date_list = pd.date_range(start= '2018-08-01', end='2018-10-31') # = 90 days
datedf = pd.DataFrame(date_list)
datedf.columns=['date']

datedf['date'] = pd.to_datetime(datedf['date'], format="%Y-%m-%d")
#add the year, month and day of the week
datedf['year'] = datedf['date'].dt.year
datedf['month'] = datedf['date'].dt.month
datedf['dayofweek'] = datedf['date'].dt.weekday

# add weekends and black friday
datedf = datedf.assign(weekend = lambda dataframe:
                dataframe['dayofweek'].map(lambda dayofweek:
                                           "Yes" if ((dayofweek == 5) | (dayofweek == 6))
                                           else "No"))
                # Notes: 5 for Saturday, 6 for Sunday since Monday is 0

datedf = datedf.assign(black_friday = lambda dataframe:
                dataframe['date'].map(lambda date:
                                           "Yes" if ((date == "2017-11-24"))
                                      else "No") )

datedf


# In[28]:


df_x = pt.build_design_matrices([t.design_info], datedf)
Pred = ModelFit.predict(df_x)
Pred=np.round(Pred)
Pred_df = pd.DataFrame(Pred.T, datedf['date'])
Pred_df.columns=['predicted_sales']
Pred_df


# In[29]:


# the predicted sum of sales for furniture and decor for Aug, Sep, and Oct 2018
predictions = Pred_df['predicted_sales'].sum()
predictions

