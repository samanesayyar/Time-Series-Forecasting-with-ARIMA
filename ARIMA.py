# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:46:44 2024

@author: Samane
"""
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings
data = pd.read_csv("Dataset/ARIMA.csv")
#print(data.head())

data=data[["Date", "Close"]]
#print(data.head())
# plt.style.use('fivethirtyeight')
# plt.figure(figsize=(15, 10))
# plt.plot(data["Date"], data["Close"])

##to distinguish our dataset whether it is stationary or non-stationary
# result = seasonal_decompose(x=data["Close"], model='multiplicative',period=4)
# result.plot().suptitle('Decompose', fontsize=22)
# plt.show()

##how to decide the p-value
ax=pd.plotting.autocorrelation_plot(data["Close"])

plt.show()

##find the value of q (moving average):

ax2=plot_pacf(data["Close"], lags = 100, method="ywm")
plt.show()

p, d, q = 5, 1, 2
# model = ARIMA(data["Close"], order=(p,d,q))  
# fitted = model.fit()  
# print(fitted.summary())
# predictions = fitted.predict()
# print(predictions)


model=sm.tsa.statespace.SARIMAX(data['Close'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())
predictions = model.predict(len(data), len(data)+10)
print(predictions)

##prediction plot
data["Close"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")