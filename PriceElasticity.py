
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[51]:


#read in csv file
df = pd.read_csv('Mercedes Price Data.csv',delimiter=';')


# #### Data pre-processing and feature engineering ####

# In[52]:


#strip leading and trailing whitespace from columns
col_name=[]
for col in df.columns:
    name = col.lstrip()
    name = name.rstrip()
    name = name.lower()
    col_name.append(name)
    
df.columns=col_name


# In[53]:


#create function to extract month value only
def extract_month(row):
    month_name = row['date'][:3]
    return month_name


# In[54]:


#create function to extract month value only
def extract_year(row):
    year = '20'+row['date'][4:]
    return year


# In[55]:


#create function to convert text date to datetime
def conv_date(row):
    months={'gen':'01','feb':'02','mar':'03','apr':'04','mag':'05','giu':'06','lug':'07','ago':'08','set':'09','ott':'10','nov':'11','dic':'12'}
    day = '01'
    month_name = row['date'][:3]
    month_val = months[month_name]
    year = '20'+row['date'][4:]
    date_time_str = year+'-'+month_val+'-'+day+' 00:00:00'
    date = datetime.strptime(date_time_str,'%Y-%m-%d %H:%M:%S')
    return date


# In[56]:


#extract month
df['month'] = df.apply(extract_month, axis = 1)
#extract month
df['year'] = df.apply(extract_year, axis = 1)
#convert text date to datetime
df['date'] = df.apply(conv_date, axis = 1)


# In[57]:


df.describe()


# In[58]:


#calculate % of nan in each column
for col in df.columns:
    perc_nan = (df[col].isnull().sum()/df[col].count())*100
    print(col+': '+str(perc_nan)+'%')


# In[59]:


#plot untis to visualize missing data points
f1, ax = plt.subplots(figsize = (30,10))
plt.plot(df['date'],df['unit.c'],'b')
plt.plot(df['date'],df['unit.e'],'c')
plt.plot(df['date'],df['unit.s'],'m')
plt.plot(df['date'],df['unit.sl'],'r')
plt.plot(df['date'],df['unit.m'],'g')


# In[60]:


#to facilitate time interpolation index must be datetime
df.set_index('date', drop=True, inplace=True)


# In[61]:


#use time series interpolation to impute missing data points
for col in ['unit.c','unit.e','unit.s','unit.cl','unit.sl','unit.m']:
    df[col].interpolate(method='time',inplace=True)


# In[62]:


#plot untis to visualize missing data points
f1, ax = plt.subplots(figsize = (30,10))
plt.plot(df['unit.c'],'b')
plt.plot(df['unit.e'],'c')
plt.plot(df['unit.s'],'m')
plt.plot(df['unit.sl'],'r')
plt.plot(df['unit.m'],'g')


# In[63]:


df.to_csv('Mercedes Price Data v2.csv')


# ### Linear Regression Model ###

# In[64]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[65]:


#normalize units to range 0 to 1
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
#split units and prices to seperate arrays
normalized_unit_df = df[['unit.c','unit.e', 'unit.s', 'unit.cl', 'unit.sl', 'unit.m']].values
normalized_price_df = df[['c.class.average.price', 'e.class.average.price', 's.class.average.price', 'cl.class.average.price', 'sl.class.average.price', 'm.class.average.price']].values
#apply normalization to arrays
normalized_unit_df = min_max_scaler.fit_transform(normalized_unit_df )
normalized_price_df = min_max_scaler.fit_transform(normalized_price_df )
#create df's from arrays
normalized_unit_df = pd.DataFrame(normalized_unit_df, index=df.index, columns=['unit.c','unit.e', 'unit.s', 'unit.cl', 'unit.sl', 'unit.m'])
normalized_price_df = pd.DataFrame(normalized_price_df, index=df.index, columns=['c.class.average.price', 'e.class.average.price', 's.class.average.price', 'cl.class.average.price', 'sl.class.average.price', 'm.class.average.price'])
#merge df's
normalized_df = pd.merge(normalized_unit_df, normalized_price_df, left_index=True, right_index=True)
normalized_df.head()


# In[66]:


#plot price vs units
sns.lmplot(x='c.class.average.price',y='unit.c',data=normalized_df)
plt.title("C Class Price vs Units")
sns.lmplot(x='e.class.average.price',y='unit.e',data=normalized_df)
plt.title("E Class Price vs Units")
sns.lmplot(x='s.class.average.price',y='unit.s',data=normalized_df)
plt.title("S Class Price vs Units")
sns.lmplot(x='cl.class.average.price',y='unit.cl',data=normalized_df)
plt.title("CL Class Price vs Units")
sns.lmplot(x='sl.class.average.price',y='unit.sl',data=normalized_df)
plt.title("SL Class Price vs Units")
sns.lmplot(x='m.class.average.price',y='unit.m',data=normalized_df)
plt.title("M Class Price vs Units")


# In[67]:


#declare x and y variables
y = df[['unit.c']]
X = df[['c.class.average.price']]


# In[68]:


#fit linear model
lm = LinearRegression()
lm.fit(X,y)


# In[69]:


# print the intercept
lm.intercept_[0]


# In[70]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[71]:


print(lm.score(X,y))


# #### Include month variable dummies to account for seasonality

# In[72]:


# create dummy variables for months
months = df[['month']]
months = pd.get_dummies(months,prefix=['month_'], drop_first=True)


# In[73]:


df = pd.merge(left = df, right = months, left_index=True, right_index=True)


# In[74]:


#declare x and y variables
y = df['unit.c']
X = df[['c.class.average.price','month__apr','month__dic', 'month__feb', 'month__gen', 'month__giu', 'month__lug','month__mag', 'month__mar', 'month__nov', 'month__ott', 'month__set']]


# In[75]:


#fit linear model
lm = LinearRegression()
lm.fit(X,y)


# In[76]:


print(lm.score(X,y))


# In[77]:


# print the intercept
lm.intercept_


# In[78]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[79]:


y_hat = lm.predict(X)


# In[80]:


df['y_hat']=lm.predict(X)


# In[81]:


#plot untis to visualize missing data points
f1, ax = plt.subplots(figsize = (30,10))
plt.plot(df['unit.c'],'b',label='Actual')
plt.plot(df['y_hat'],'r',linestyle='--',label='Predicted')
plt.legend()


# In[82]:


# Plot the residuals of linear model
sns.residplot(y_hat, y_hat-y, lowess=True, color="g")
plt.title("C Class Price Residual Plot")
plt.ylabel('Residuals')


# Residual plot does not show any specific trend to indicate issues with the model, possibly increased variance, will try applying transformation to variables

# #### Log Transformation of Variables - Log : Log model ####

# In[83]:


#apply log transformation to price and dependent variables
log_df = df[['c.class.average.price','unit.c']]
log_df['c.class.average.price(log)'] = log_df['c.class.average.price'].apply(np.log)


# In[84]:


#merge transformed data with monthly dummy variables
log_df = pd.merge(left = log_df, right = months, left_index=True, right_index=True)


# In[85]:


#declare x and y variables
y = log_df['unit.c'].apply(np.log)
X = log_df[['c.class.average.price(log)','month__apr','month__dic', 'month__feb', 
        'month__gen', 'month__giu', 'month__lug','month__mag', 
        'month__mar', 'month__nov', 'month__ott', 'month__set']]


# In[86]:


#fit linear model
lm = LinearRegression()
lm.fit(X,y)


# In[87]:


#print score
print(lm.score(X,y))


# In[88]:


# print the intercept
lm.intercept_


# In[89]:


#create df of model coefficients and intercept
coeff_labels = X.columns[1:]
coeff_labels = coeff_labels.insert(0,'Log Avg Price')
coeff_labels = coeff_labels.insert(0,'Intercept')
coeff_vals = lm.coef_.tolist()
coeff_vals.insert(0, lm.intercept_)
coeff_vals
coeffs = pd.DataFrame(dict(zip(coeff_labels, coeff_vals)),index=['C Class Coeffs'])
coeffs


# In[90]:


y_hat = lm.predict(X)
log_df['y_hat']=lm.predict(X)


# In[91]:


#plot predictions vs actual
f1, ax = plt.subplots(figsize = (30,10))
plt.plot(log_df['unit.c'].apply(np.log),'b',label='Actual')
plt.plot(log_df['y_hat'],'r',marker='*',linestyle='--',label='Predicted')
plt.legend()


# In[92]:


# Plot the residuals of linear model
sns.residplot(y_hat, y_hat-y, lowess=True, color="g")
plt.title("C Class Price Residual Plot")
plt.ylabel('Residuals')


# slight positive trend in residual variance towards right, but mostly centred around 0

# #### log lin model ####

# In[93]:


#create data set
log_df = df[['c.class.average.price','unit.c']]


# In[94]:


#merge with monthly dummy variables
log_df = pd.merge(left = log_df, right = months, left_index=True, right_index=True)


# In[95]:


#declare x and y variables
y = log_df['unit.c'].apply(np.log)
X = log_df[['c.class.average.price','month__apr','month__dic', 'month__feb', 
        'month__gen', 'month__giu', 'month__lug','month__mag', 
        'month__mar', 'month__nov', 'month__ott', 'month__set']]


# In[96]:


#fit linear model
lm = LinearRegression()
lm.fit(X,y)


# In[97]:


#print score
print(lm.score(X,y))


# In[98]:


# print the intercept
lm.intercept_


# In[99]:


#create df of coeffs and intercept
coeff_labels = X.columns[1:]
coeff_labels = coeff_labels.insert(0,'Avg Price')
coeff_labels = coeff_labels.insert(0,'Intercept')
coeff_vals = lm.coef_.tolist()
coeff_vals.insert(0, lm.intercept_)
coeff_vals
coeffs = pd.DataFrame(dict(zip(coeff_labels, coeff_vals)),index=['C Class Coeffs'])
coeffs


# In[100]:


#include predictions in df
log_df['y_hat']=lm.predict(X)


# In[101]:


#plot predictions vs actual
f1, ax = plt.subplots(figsize = (30,10))
plt.plot(log_df['unit.c'].apply(np.log),'b',label='Actual')
plt.plot(log_df['y_hat'],'r',marker='*',linestyle='--',label='Predicted')
plt.legend()


# #### lin log model ####

# In[102]:


#create df and apply log transform to price only
log_df = df[['c.class.average.price','unit.c']]
log_df['c.class.average.price(log)'] = log_df['c.class.average.price'].apply(np.log)


# In[103]:


#merge df with monthly dummys
log_df = pd.merge(left = log_df, right = months, left_index=True, right_index=True)


# In[104]:


#declare x and y variables
y = log_df['unit.c']
X = log_df[['c.class.average.price(log)','month__apr','month__dic', 'month__feb', 
        'month__gen', 'month__giu', 'month__lug','month__mag', 
        'month__mar', 'month__nov', 'month__ott', 'month__set']]


# In[105]:


#fit linear model
lm = LinearRegression()
lm.fit(X,y)


# In[106]:


#print score
print(lm.score(X,y))


# In[107]:


# print the intercept
lm.intercept_


# In[108]:


#create df of coeffs and intercept
coeff_labels = X.columns[1:]
coeff_labels = coeff_labels.insert(0,'Log Avg Price')
coeff_labels = coeff_labels.insert(0,'Intercept')
coeff_vals = lm.coef_.tolist()
coeff_vals.insert(0, lm.intercept_)
coeff_vals
coeffs = pd.DataFrame(dict(zip(coeff_labels, coeff_vals)),index=['C Class Coeffs'])
coeffs


# In[109]:


#include predcitions in df
log_df['y_hat']=lm.predict(X)


# In[110]:


#plot predictions vs actual
f1, ax = plt.subplots(figsize = (30,10))
plt.plot(log_df['unit.c'],'b',label='Actual')
plt.plot(log_df['y_hat'],'r',marker='*',linestyle='--',label='Predicted')
plt.legend()


# #### Exp transformation of price ###

# In[111]:


exp_df = df[['c.class.average.price','unit.c']]


# In[112]:


exp_df = pd.merge(left = exp_df, right = months, left_index=True, right_index=True)


# In[113]:


exp_df['c.class.average.price'] = exp_df['c.class.average.price']**2


# In[114]:


#declare x and y variables
y = exp_df['unit.c']
X = exp_df.drop(columns=['unit.c'])


# In[115]:


#fit linear model
lm = LinearRegression()
lm.fit(X,y)


# In[116]:


#print score
print(lm.score(X,y))


# In[117]:


# print the intercept
lm.intercept_


# In[118]:


coeff_labels = X.columns[1:]
coeff_labels = coeff_labels.insert(0,'Exp Avg Price')
coeff_labels = coeff_labels.insert(0,'Intercept')
coeff_vals = lm.coef_.tolist()
coeff_vals.insert(0, lm.intercept_)
coeff_vals
coeffs = pd.DataFrame(dict(zip(coeff_labels, coeff_vals)),index=['C Class Coeffs'])
coeffs


# In[119]:


y_hat = lm.predict(X)
exp_df['y_hat']=lm.predict(X)


# In[120]:


#plot predictions vs actual
f1, ax = plt.subplots(figsize = (30,10))
plt.plot(log_df['unit.c'],'b',label='Actual')
plt.plot(log_df['y_hat'],'r',marker='*',linestyle='--',label='Predicted')
plt.legend()


# ### Time Series Forecast ###

# In[121]:


#import required libraries
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller


# In[122]:


#create df of units variable
forecast_df = df[['unit.c']]


# In[123]:


#plot untis over time
f1, ax = plt.subplots(figsize = (30,10))
plt.plot(forecast_df['unit.c'],'b')


# In[124]:


#perform augmented dickey fuller test for differencing
result = adfuller(forecast_df['unit.c'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# Dickey Fuller test indicates stationary variable, can be further illustrated with autocorrelation plot

# In[125]:


from pandas.tools.plotting import autocorrelation_plot
f1, ax = plt.subplots(figsize = (30,10))
autocorrelation_plot(forecast_df)


# In[126]:


#loop through order variables to find the lowest AIC score
AIC_results = pd.DataFrame(columns = ['p','d','q','AIC'],index=[])
i=0
for p in [0,1,2,3,4,5]:
    for q in [0,1,2,3,4,5]:
        for d in [0,1,2]:
            try:
                model = ARIMA(forecast_df,order=(p,d,q))
                model_fit = model.fit(disp=0)
            except:
                line_result = pd.DataFrame({'p':p,'d':d,'q':q,'AIC':np.nan},index=[i])
            else:
                line_result = pd.DataFrame({'p':p,'d':d,'q':q,'AIC':model_fit.aic},index=[i])
            AIC_results = AIC_results.append(line_result)
            i +=1


# In[127]:


#display top AIC scores
AIC_results.sort_values('AIC',ascending=True).head()


# In[128]:


model = ARIMA(forecast_df, order=(1,2,4))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[146]:


#plot residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
residuals.plot(kind='kde')


# In[147]:


#plot predcited against actual
model_fit.plot_predict()


# In[150]:


#create training test split (80:20)
stop = int(len(forecast_df)*.8)
train = forecast_df[:stop]
test = forecast_df[stop:]


# In[151]:


#function to calculate the mean squared error
def calc_mse(test, f_cast):
    f_cast = pd.DataFrame(f_cast, columns=['Forecast'],index=test.index)
    f_cast = pd.merge(test,f_cast,right_index=True,left_index=True)
    f_cast['SqErr'] = (f_cast['unit.c']-f_cast['Forecast'])**2
    return f_cast['SqErr'].mean()


# In[152]:


model = ARIMA(train,order=(2,1,2))
model_fit = model.fit(disp=0)
f_cast, stderr, conf = model_fit.forecast(steps = len(test),alpha=0.05)


# In[153]:


#loop through order variables to find the best fit on test data
MSE_results = pd.DataFrame(columns = ['p','d','q','MSE'],index=[])
i=0
for p in [0,1,2,3,4,5]:
    for q in [0,1,2,3,4,5]:
        for d in [0,1,2]:
            try:
                model = ARIMA(train,order=(p,d,q))
                model_fit = model.fit(disp=0)
                f_cast, stderr, conf = model_fit.forecast(steps = len(test),alpha=0.05)
                mse = calc_mse(test, f_cast)
            except:
                line_result = pd.DataFrame({'p':p,'d':d,'q':q,'MSE':np.nan},index=[i])
            else:
                line_result = pd.DataFrame({'p':p,'d':d,'q':q,'MSE':mse},index=[i])
            MSE_results = MSE_results.append(line_result)
            i +=1


# In[154]:


MSE_results.sort_values('MSE',ascending=True).head(10)


# In[161]:


model = ARIMA(forecast_df,order=(3,0,2))
model_fit = model.fit(disp=0)
f_cast, stderr, conf = model_fit.forecast(steps = len(test),alpha=0.05)


# In[165]:


#create df from forecasted units and actual
f_cast = pd.DataFrame(f_cast, columns=['Forecast'],index=test.index)
f_cast = pd.merge(test,f_cast,right_index=True,left_index=True)


# In[166]:


#plot actual and forecast
f_cast.plot()


# In[170]:


#create forecast of next 12 months using model above
forecast = model_fit.forecast(steps = 12, alpha=0.15)


# In[171]:


#create date series of next 12 months
date_series = []
for x in [4,5,6,7,8,9,10,11,12,1,2,3]:
    if x < 4:
        year=2010
    else:
        year = 2009
    date_time_str = str(year)+'-'+str(x)+'-'+'1'+' 00:00:00'
    date = datetime.strptime(date_time_str,'%Y-%m-%d %H:%M:%S')
    date_series.append(date)


# In[172]:


#create df of model forecast outputs and date series
forecast_12m = pd.DataFrame({'Forecast':forecast[0],'StdErr':forecast[1],'ConfIntLow':forecast[2][:,0],'ConfIntHigh':forecast[2][:,1]},index=date_series)


# In[173]:


forecast_12m.plot()


# In[174]:


#write forecast to csv
forecast_12m.to_csv('forecast_12m.csv')

