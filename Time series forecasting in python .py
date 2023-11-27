#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns 
from matplotlib import pyplot as plt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression 
import warnings
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import month_plot


# In[2]:


df = pd.read_csv("C://Users//Kiruthicka//Downloads//monthly_csv.csv")
df.head()


# In[3]:


df.shape


# In[4]:


print(f"Date range of gold prices available from - {df.loc[:, 'Date'][0]} to {df.loc[:, 'Date'][len(df)-1]}")


# In[5]:


date=pd.date_range(start = '1/1/1950',end='8/1/2020',freq ='M')
date


# In[6]:


df["month"]=date
df.drop('Date',axis=1,inplace=True)
df=df.set_index('month')
df.head()


# In[7]:


df.plot(figsize=(20,8))
plt.title("Gold prices monthly since 1950 and onwards")
plt.xlabel("months")
plt.ylabel("price")
plt.grid();


# In[8]:


round(df.describe(),3)


# In[9]:


_, ax = plt.subplots(figsize=(25,8))
sns.boxplot(x=df.index.year , y = df.values[:,0],ax=ax)
plt.title('Gold price monthly since 1950 onwards')
plt.xlabel('year')
plt.ylabel('price')
plt.xticks(rotation = 90)
plt.grid();


# In[10]:


fig,ax =plt.subplots(figsize=(22,8))
month_plot(df,ylabel="gold price",ax=ax)
plt.title('Gold price monthly since 1950 onwards')
plt.xlabel('month')
plt.ylabel('price')
plt.grid();


# In[11]:


_,ax = plt.subplots(figsize=(22,8))
sns.boxplot(x=df.index.month_name(),y=df.values[:,0],ax=ax)
plt.title('Gold price monthly since 1950 onwards')
plt.xlabel("month")
plt.ylabel("price")
plt.show()


# In[12]:


df_yearly_sum = df.resample('A').mean()
df_yearly_sum.plot();
plt.title("Average gold price yearly since 1950")
plt.xlabel('year')
plt.ylabel('price')
plt.grid();


# In[13]:


df_quarterly_sum=df.resample('Q').mean()
df_quarterly_sum.plot();
plt.title("Average gold price yearly since 1950")
plt.xlabel('Quarter')
plt.ylabel('price')
plt.grid();


# In[14]:


df_decade_sum = df.resample('10Y').mean()
df_decade_sum.plot();
plt.title('Average gold price for decade since 1950')
plt.xlabel('Decade')
plt.ylabel('price')
plt.grid();


# In[15]:


df_1 = df.groupby(df.index.year).agg({'Price': ['mean', 'std']})
df_1.columns = ['Mean', 'Std']
df_1['Cov_pct'] = ((df_1['Std'] / df_1['Mean']) * 100).round(2)
df_1.head()


# In[16]:


fig, ax = plt.subplots(figsize=(15, 10))
df_1['Cov_pct'].plot(ax=ax)
plt.title("Average gold price yearly since 1950")
plt.xlabel('year')
plt.show()


# In[17]:


train=df[df.index.year<=2015]
test =df[df.index.year>2015]


# In[18]:


print(train.shape)


# In[19]:


print(test.shape)


# In[20]:


train["Price"].plot(figsize=(13,5),fontsize=15)
test["Price"].plot(figsize=(13,5),fontsize=15)
plt.grid()
plt.legend(['Training Data' , 'Test Data'])
plt.show()


# In[21]:


train_time=[i+1 for i in range(len(train))]
test_time=[i+len(train)+1 for i in range(len(test))]
len(train_time),len(test_time)


# In[22]:


LR_train = train.copy()
LR_test = test.copy()


# In[23]:


LR_train['time']=train_time
LR_test['time']=test_time


# In[24]:


lr = LinearRegression()
lr.fit(LR_train[['time']],LR_train["Price"].values)


# In[25]:


test_predictions_model1 = lr.predict(LR_test[["time"]])
LR_test['forecast']=test_predictions_model1

plt.figure(figsize=(14, 6))
plt.plot(train['Price'],label='train')
plt.plot(test['Price'],label='test')
plt.plot(LR_test['forecast'],label='reg on time_test data')
plt.legend(loc ='best')
plt.grid();


# In[26]:


def mape(actual,pred):
    return round((np.mean(abs(actual-pred)/actual))*100,2)


# In[27]:


mape_model1_test =mape(test['Price'].values,test_predictions_model1)
print("MAPE is %3.3f" % (mape_model1_test), "%")


# In[28]:


results=pd.DataFrame({'Test Mape(%)':[mape_model1_test]},index=['RegressionOnTime'])
results


# In[29]:


Naive_train=train.copy()
Naive_test=test.copy()


# In[30]:


Naive_test['naive']=np.asarray(train['Price'])[len(np.asarray(train['Price']))-1]
Naive_test['naive'].head()


# In[31]:


plt.figure(figsize=(12,8))
plt.plot(Naive_train['Price'],label='Train')
plt.plot(test['Price'],label='Test')
plt.plot(Naive_test['naive'],label='Naive Forecast on Test Data')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.grid();


# In[32]:


mape_model2_test=mape(test['Price'].values,Naive_test['naive'].values)
print("For Naive forecast on the Test Data, MAPE is %3.3f" % mape_model2_test, "%")


# In[33]:


resultsDf_2 = pd.DataFrame({'Test MAPE (%)':[mape_model2_test]},index=['NaiveModel'])
results = pd.concat([results,resultsDf_2])
results


# In[34]:


final_model = ExponentialSmoothing(df,
                                   trend='additive',  
                                   seasonal='additive').fit(smoothing_level=0.4,
                                                          smoothing_trend=0.3,
                                                          smoothing_seasonal=0.6)


# In[35]:


Mape_final_model = mape(df['Price'].values, final_model.fittedvalues)
print("MAPE:", Mape_final_model)


# In[36]:


predictions  = final_model.forecast(steps=len(test))


# In[37]:


pred_df =pd.DataFrame({'lower_CI':predictions -1.96*np.std(final_model.resid,ddof=1),
                         'prediction':predictions,
                         'upper_CI':predictions+1.96*np.std(final_model.resid,ddof=1)})
pred_df.head()


# In[38]:


axis = df.plot(label='Actual', figsize=(15, 9))
pred_df['prediction'].plot(ax=axis,label ='Forecast',alpha =0.5)
axis.fill_between(pred_df.index,pred_df['lower_CI'] ,pred_df['upper_CI'],color='m',alpha=.15)
axis.set_xlabel('year_month')
axis.set_ylabel('price')
plt.legend(loc='best')
plt.grid()
plt.show()


# In[ ]:




