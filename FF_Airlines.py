#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries <a name="Import-Libraries"></a>

# In[13]:


from datetime import datetime
import numpy as np            
import pandas as pd            
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6


# In[14]:


Airlines = pd.read_excel('C:/Users/prate/Downloads/Assignment/Forecasting/Airlines_Data.xlsx')
Airlines


# In[15]:


Airlines['Month'] = pd.to_datetime(Airlines['Month'],infer_datetime_format=True) #convert from string to datetime
Airlines = Airlines.set_index(['Month'])
Airlines.head(5)


# From the plot below, we can see that there is a Trend compoenent in th series. Hence, we now check for stationarity of the data

# In[16]:


## plot graph
plt.xlabel('Date')
plt.ylabel('Number of air passengers')
plt.plot(Airlines)


# In[17]:


#Determine rolling statistics
rolmean = Airlines.rolling(window=12).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level
rolstd = Airlines.rolling(window=12).std()
print(rolmean,rolstd)


# In[18]:


#Plot rolling statistics
orig = plt.plot(Airlines, color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# In[19]:


#Perform Augmented Dickey–Fuller test:
print('Results of Dickey Fuller Test:')
dftest = adfuller(Airlines['Passengers'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    
print(dfoutput)


# For a Time series to be stationary, its ADCF test should have:
# 1. p-value to be low (according to the null hypothesis)
# 2. The critical values at 1%,5%,10% confidence intervals should be as close as possible to the Test Statistics
# 
# From the above ADCF test result, we see that p-value(at max can be 1.0) is very large. Also critical values are no where close to the Test Statistics. Hence, we can safely say that **our Time Series at the moment is not stationary**

# ## Data Transformation to achieve Stationarity <a name="data-transform"></a>
# 
# There are a couple of ways to achieve stationarity through data transformation like taking $log_{10}$,$log_{e}$, square, square root, cube, cube root, exponential decay, time shift and so on ...
# 
# In our notebook, lets start of with log transformations. Our objective is to remove the trend component. Hence,  flatter curves( ie: paralle to x-axis) for time series and rolling mean after taking log would say that our data transformation did a good job.

# ### Log Scale Transformation  <a name="log"></a>

# In[22]:


#Estimating trend
Airlines_logScale = np.log(Airlines)
plt.plot(Airlines_logScale)
Airlines.dropna()


# In[23]:


#The below transformation is required to make series stationary
movingAverage = Airlines_logScale.rolling(window=12).mean()
movingSTD = Airlines_logScale.rolling(window=12).std()
plt.plot(Airlines_logScale)
plt.plot(movingAverage, color='red')


# In[24]:


datasetLogScaleMinusMovingAverage = Airlines_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#Remove NAN values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)


# In[25]:


def test_stationarity(timeseries):
    
    #Determine rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    #Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


# In[26]:


test_stationarity(datasetLogScaleMinusMovingAverage)


# ### Exponential Decay Transformation   <a name="exp"></a>

# In[27]:


exponentialDecayWeightedAverage = Airlines_logScale.ewm(halflife=4, min_periods=0, adjust=True).mean()
plt.plot(Airlines_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')


# In[28]:


datasetLogScaleMinusExponentialMovingAverage = Airlines_logScale - exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusExponentialMovingAverage)


# ### Time Shift Transformation  <a name="shift"></a>

# In[29]:


datasetLogDiffShifting = Airlines_logScale - Airlines_logScale.shift()
plt.plot(datasetLogDiffShifting)


# In[30]:


datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)


# In[31]:


decomposition = seasonal_decompose(Airlines_logScale) 

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(Airlines_logScale, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(411)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')

plt.subplot(411)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')

plt.tight_layout()

#there can be cases where an observation simply consisted of trend & seasonality. In that case, there won't be 
#any residual component & that would be a null or NaN. Hence, we also remove such cases.
decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[32]:


decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# ## Plotting ACF & PACF <a name="acf-pacf"></a>

# In[33]:


#ACF & PACF plots

lag_acf = acf(datasetLogDiffShifting, nlags=10)
lag_pacf = pacf(datasetLogDiffShifting, nlags=10, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')            

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
            
plt.tight_layout()            


# ## Building Models <a name="model"></a>

# In[36]:


#AR Model
#making order=(2,1,0) 
model = ARIMA(Airlines_logScale, order=(2,1,0))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_AR.fittedvalues - datasetLogDiffShifting['Passengers'])**2))
print('Plotting AR model')


# In[37]:


#MA Model
model = ARIMA(Airlines_logScale, order=(0,1,2))
results_MA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_MA.fittedvalues - datasetLogDiffShifting['Passengers'])**2))
print('Plotting MA model')


# In[38]:


# AR+I+MA = ARIMA model
model = ARIMA(Airlines_logScale, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['Passengers'])**2))
print('Plotting ARIMA model')


# #By combining AR & MA into ARIMA, we see that RSS value has decreased from either case to 0.3911, indicating ARIMA to be better #than its individual component models.   
# 
# #With the ARIMA model built, we will now generate predictions. But, before we do any plots for predictions ,we need to reconvert #the predictions back to original form. This is because, our model was built on log transformed data.

# ## Prediction & Reverse transformations <a name="prediction"></a>

# In[39]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())


# In[40]:


#Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum)


# In[43]:


predictions_ARIMA_log = pd.Series(Airlines_logScale['Passengers'].iloc[0], index=Airlines_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()


# In[44]:


# Inverse of log is exp.
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(Airlines)
plt.plot(predictions_ARIMA)


# We see that our predicted forecasts are very close to the real time series values indicating a fairly accurate model.

# In[45]:


Airlines_logScale


# In[46]:


results_ARIMA.plot_predict(1,216) 


# In[ ]:




