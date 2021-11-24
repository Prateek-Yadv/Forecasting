#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries <a name="Import-Libraries"></a>

# In[136]:


from datetime import datetime
import numpy as np             #for numerical computations like log,exp,sqrt etc
import pandas as pd            #for reading & storing data, pre-processing
import matplotlib.pylab as plt #for visualization
#for making sure matplotlib plots are generated in Jupyter notebook itself
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6


# In[137]:


Cola = pd.read_excel('C:/Users/prate/Downloads/Assignment/Forecasting/CocaCola_Sales_Rawdata.xlsx')


# In[138]:


temp = Cola.Quarter.str.replace(r'(Q\d)_(\d+)', r'19\2-\1')
Cola['quarter'] = pd.to_datetime(temp).dt.strftime('%b-%Y')


# In[139]:


Cola = Cola.drop(['Quarter'],axis=1)
Cola.reset_index(inplace=True)
Cola['quarter'] = pd.to_datetime(Cola['quarter'])
Cola = Cola.set_index('quarter')
Cola.drop(["index"],axis=1,inplace=True)
Cola.head()


# From the plot below, we can see that there is a Trend compoenent in th series. Hence, we now check for stationarity of the data

# In[140]:


## plot graph
plt.xlabel('Time')
plt.ylabel('sales')
plt.plot(Cola)


# In[141]:


#Determine rolling statistics
rolmean = Cola.rolling(window=4).mean() #window size 4 denotes 4 quarters, giving rolling mean at yearly level
rolstd = Cola.rolling(window=4).std()
print(rolmean,rolstd)


# In[142]:


#Plot rolling statistics
orig = plt.plot(Cola, color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# In[143]:


#Perform Augmented Dickey–Fuller test:
print('Results of Dickey Fuller Test:')
dftest = adfuller(Cola['Sales'], autolag='AIC')

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

# In[144]:


#Estimating trend
Cola_logScale = np.log(Cola)
plt.plot(Cola_logScale)
Cola.dropna()


# In[145]:


#The below transformation is required to make series stationary
movingAverage = Cola_logScale.rolling(window=4).mean()
movingSTD = Cola_logScale.rolling(window=4).std()
plt.plot(Cola_logScale)
plt.plot(movingAverage, color='red')


# In[146]:


datasetLogScaleMinusMovingAverage = Cola_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#Remove NAN values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)


# In[147]:


def test_stationarity(timeseries):
    
    #Determine rolling statistics
    movingAverage = timeseries.rolling(window=4).mean()
    movingSTD = timeseries.rolling(window=4).std()
    
    #Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


# In[148]:


test_stationarity(datasetLogScaleMinusMovingAverage)


# ### Exponential Decay Transformation   <a name="exp"></a>

# In[149]:


exponentialDecayWeightedAverage = Cola_logScale.ewm(halflife=4, min_periods=0, adjust=True).mean()
plt.plot(Cola_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')


# In[150]:


datasetLogScaleMinusExponentialMovingAverage = Cola_logScale - exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusExponentialMovingAverage)


# ### Time Shift Transformation  <a name="shift"></a>

# In[151]:


datasetLogDiffShifting = Cola_logScale - Cola_logScale.shift()
plt.plot(datasetLogDiffShifting)


# In[152]:


datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)


# In[153]:


decomposition = seasonal_decompose(Cola_logScale) 

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(Cola_logScale, label='Original')
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


# In[154]:


decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# ## Plotting ACF & PACF <a name="acf-pacf"></a>

# In[155]:


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

# In[156]:


#AR Model
#making order=(2,1,0) 
model = ARIMA(Cola_logScale, order=(2,1,0))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_AR.fittedvalues - datasetLogDiffShifting['Sales'])**2))
print('Plotting AR model')


# In[157]:


#MA Model
model = ARIMA(Cola_logScale, order=(0,1,2))
results_MA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_MA.fittedvalues - datasetLogDiffShifting['Sales'])**2))
print('Plotting MA model')


# In[158]:


# AR+I+MA = ARIMA model
model = ARIMA(Cola_logScale, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['Sales'])**2))
print('Plotting ARIMA model')


# #By combining AR & MA into ARIMA, we see that RSS value has decreased from either case to 0.3911, indicating ARIMA to be better #than its individual component models.   
# 
# #With the ARIMA model built, we will now generate predictions. But, before we do any plots for predictions ,we need to reconvert #the predictions back to original form. This is because, our model was built on log transformed data.

# ## Prediction & Reverse transformations <a name="prediction"></a>

# In[159]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())


# In[160]:


#Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum)


# In[161]:


predictions_ARIMA_log = pd.Series(Cola_logScale['Sales'].iloc[0], index=Cola_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()


# In[162]:


# Inverse of log is exp.
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(Cola)
plt.plot(predictions_ARIMA)


# We see that our predicted forecasts are very close to the real time series values indicating a fairly accurate model.

# In[163]:


Cola_logScale


# In[164]:


results_ARIMA.plot_predict(1,80) 


# In[ ]:




