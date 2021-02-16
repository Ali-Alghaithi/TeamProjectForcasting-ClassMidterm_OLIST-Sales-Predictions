#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# OLS for Bed and Bath


# In[1]:


import pandas as pd
import numpy as np
import datetime
import plotly.express as px


# In[2]:


# Bed_Bath Category


# In[3]:


# Load the data
bed_bath = pd.read_csv('bed_bath_table_data.csv')

bb_categ = bed_bath
# define short column names
bb_categ.columns = ['date', 'category', 'volume', 'revenue']
# Changing date data type to datetime
bb_categ['date'] = pd.to_datetime(bb_categ['date'], format="%Y-%m-%d")
#add the year, month and day of the week
bb_categ['year'] = bb_categ['date'].dt.year
bb_categ['month'] = bb_categ['date'].dt.month
bb_categ['dayofweek'] = bb_categ['date'].dt.weekday

# add weekends and black friday
bb_categ = bb_categ.assign(weekend = lambda dataframe:
                dataframe['dayofweek'].map(lambda dayofweek:
                                           "Yes" if ((dayofweek == 5) | (dayofweek == 6))
                                           else "No"))

# Notes: 5 for Saturday, 6 for Sunday since Monday is 0
'''bb_categ = bb_categ.assign(black_friday = lambda dataframe:
                dataframe['date'].map(lambda date:
                                           "Yes" if ((date == "2017-11-24"))
                                           else "No") )'''

bb_categ = bb_categ.assign(black_friday = lambda dataframe:
                dataframe['volume'].map(lambda volume:
                                           "Yes" if ((volume == max(bb_categ['volume'])))
                                           else "No"))
print(bb_categ.head(), "\n")
print(bb_categ.tail())


# In[ ]:





# In[5]:


# plotting volume of sales
px.line(x=bb_categ['date'], y=bb_categ['volume'])


# In[8]:


# Split the data into training and  sets
train = bb_categ[:474]
test = bb_categ[474:]
test.head(3)


# In[15]:


# building the model
import patsy as pt
y, x = pt.dmatrices("volume ~ C(year) + C(month) : C(dayofweek) + black_friday + weekend",
                   data = train)

import statsmodels.api as sm
# Declare the model, and create an instance of the OLS
#  class
model = sm.OLS(endog = y, exog = x)
# Fit the model, optionally using specific parameters
modelFit = model.fit()
modelFit.summary()


# In[16]:


# Residuals
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
res = modelFit.resid
res = pd.Series(res)
ax = res.plot.kde()
ax # The errors are normally distributed with mean 0


# In[17]:


# Prediction using test data
test_x = pt.build_design_matrices([x.design_info], test)
pred = modelFit.predict(test_x).squeeze()

# plot test actual and predicted
px.line(test, x='date', y=['volume', pred]) # seems acceptable


# In[18]:


# MSE
from sklearn.metrics import mean_squared_error 
MSE = mean_squared_error(test['volume'], pred)
print('MSE = ', MSE)


# In[205]:


# USING THE WHOLE DATASET TO BUILD THE MODEL AND MAKE THE  MONTHS PREDICTIONS


# In[268]:


# building the model
import patsy as pt
y, x = pt.dmatrices("volume ~ C(year) + C(month) + C(dayofweek) + black_friday + weekend",
                   data = bb_categ)

import statsmodels.api as sm
# Declare the model, and create an instance of the OLS
#  class
Model = sm.OLS(endog = y, exog = t)
# Fit the model, optionally using specific parameters
ModelFit = Model.fit()
ModelFit.summary()


# In[269]:


# Residuals
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
resid = ModelFit.resid
resid = pd.Series(resid)
ax = resid.plot.kde()
ax


# In[237]:


# Prediction using rthe whole dataset
Pred = ModelFit.predict(t).squeeze()

# plot  actual and predicted
px.line(bb_categ, x='date', y=['volume', Pred])


# In[ ]:





# In[211]:


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


# In[212]:


df_x = pt.build_design_matrices([t.design_info], datedf)
Pred = ModelFit.predict(df_x)
Pred=np.round(Pred)
Pred_df = pd.DataFrame(Pred.T, datedf['date'])
Pred_df.columns=['predicted_sales']
Pred_df


# In[213]:


# the predicted sum of sales for bath and bed for Aug, Sep, and Oct 2018
sales_3m = Pred_df['predicted_sales'].sum()
sales_3m


# In[ ]:




