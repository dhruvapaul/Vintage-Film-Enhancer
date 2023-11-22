#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Dhruva Paul
#Stock Prediction Final CS Project
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.losses import MeanSquaredError


# In[2]:


#read in aluminum prices as data, take only the last 5 years 
aluminum = pd.read_csv("alumnium.csv")
aluminum = aluminum.iloc[330:]


# In[3]:


#read in iron prices as data, take only the last 5 years 
iron = pd.read_csv("ironn.csv")
iron = iron.iloc[330:]


# In[4]:


#read in copper prices as data, take only the last 5 years 
copper = pd.read_csv("copper.csv")
copper = copper.iloc[330:]


# In[5]:


#combine all 3 data into one dataframe
mat = aluminum.join(iron.set_index('DATE'), on="DATE")
materials = mat.join(copper.set_index('DATE'), on="DATE")
materials = materials.rename({"PALUMUSDM": "aluminum", "PIORECRUSDM": "iron", "PCOPPUSDM": "copper"}, axis='columns')
materials["DATE"] = pd.to_datetime(materials["DATE"])
materials


# In[6]:


#read in NASDAQ, Apple, Amazon, and Netflix stock prices going back 5 years, join them together
nasdaq = pd.read_csv("NASDAQ_5Y.csv")
nasdaq= nasdaq.iloc[:,[0,1]]
aapl = pd.read_csv("aaplstock.csv")
aapl = aapl.iloc[:,[0,1]]
amazon = pd.read_csv("amazon.csv")
netflix = pd.read_csv('netflix.csv')
amazon = amazon.iloc[:,[0,1]].rename({"Close/Last":'AMZN'}, axis='columns')
netflix = netflix.iloc[:,[0,1]].rename({"Close/Last":'NFLX'}, axis='columns')
stocks = nasdaq.merge(aapl, on='Date').merge(amazon, on='Date').merge(netflix, on='Date')
stocks = stocks.rename({"Close/Last_x": "NASDAQ", "Close/Last_y": "AAPL", "Date":'DATE'}, axis='columns')
stocks = stocks.reindex(index=stocks.index[::-1])
stocks


# In[7]:


#clean up data, convert String values to numeric values that can be operated on
stocks['AAPL'] = stocks['AAPL'].str.replace("$", "")
stocks['AMZN'] = stocks['AMZN'].str.replace("$", "")
stocks['NFLX'] = stocks['NFLX'].str.replace("$", "")
stocks['NASDAQ'] = stocks['NASDAQ'].str.replace("$", "")

stocks['AAPL']=stocks['AAPL'].astype(float)
stocks['NFLX']=stocks['NFLX'].astype(float)
stocks['AMZN']=stocks['AMZN'].astype(float)
stocks['NASDAQ']=stocks['NASDAQ'].astype(float)

stocks


# In[8]:


#synchronize date format with other dataframes
stocks["DATE"] = pd.to_datetime(stocks["DATE"])
stocks['DATE'] = pd.to_datetime(stocks['DATE'], format='%y%m%d')
stocks


# In[9]:


#read in revenue, gross profit, profit margin, and return on equity data, clean up data and convert String values to 
#numeric values that can be operated on
revenue = pd.read_csv("revenue - Sheet1.csv")
grossprofit = pd.read_csv("grossprofit - Sheet1 (1).csv")
profitmargin = pd.read_csv("profitmargin - Sheet1.csv")
returnonequity = pd.read_csv("returnonequity - Sheet1 (1).csv")
financials = revenue.merge(grossprofit, on='DATE').merge(profitmargin, on='DATE').merge(returnonequity, on='DATE')
financials['Return on Equity'] = financials['Return on Equity'].str.rstrip('%').astype('float') / 100.0
financials['Profit Margin'] = financials['Profit Margin'].str.rstrip('%').astype('float') / 100.0
financials['Gross Profit (in millions)'] = financials['Gross Profit (in millions)'].str.replace("$", "")
financials = financials.replace(',','', regex=True)
financials['Gross Profit (in millions)'] = financials['Gross Profit (in millions)'].astype(float)
financials["DATE"] = pd.to_datetime(financials["DATE"])
financials


# In[10]:


#combine all 3 large datasets into one final dataset through shared Date values
big = financials.merge(stocks, how = 'outer', on= 'DATE')
final = big.merge(materials, how = "outer", on="DATE")
final = final.sort_values(by='DATE')
final = final.reset_index(drop=True)
final.head(30)


# In[11]:


#interpolate between known data points, as each data section is released on different periods
data = final.interpolate(method="pchip")
data.head(30)


# In[12]:


#since we are predicting the Apple stock price of the NEXT day, we shift the column values up one 
data["AAPL"] = data["AAPL"].shift(-1)
data.dropna(inplace = True)
data.head(50)


# In[13]:


#split data into independent variables and dependent variable
X = data.loc[:,["Revenue (in billions)", "Gross Profit (in millions)", "Profit Margin", "Return on Equity", "NASDAQ", "AMZN", "NFLX", "aluminum", "iron", "copper"]]
y = data["AAPL"]
X


# In[14]:


#split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# In[15]:


#scale the data since there are values ranging from 0 to 1 as well as 3000 to 10000

def scale_datasets(X_train, X_test):
    standard_scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
          standard_scaler.fit_transform(X_train),
          columns=X_train.columns
      )
    X_test_scaled = pd.DataFrame(
          standard_scaler.transform(X_test),
          columns = X_test.columns
      )
    return X_train_scaled, X_test_scaled
X_train_scaled, X_test_scaled = scale_datasets(X_train, X_test)

#implement callbacks, which stop training the model when loss is not decreasing (saves time)
from keras.callbacks import EarlyStopping, ModelCheckpoint
callback = EarlyStopping(
    monitor="loss",
    min_delta=0,
    patience=3,
    verbose=1,
    mode="min",
)


# In[16]:


hidden_units1 = 256
hidden_units2 = 256
hidden_units3 = 256
learning_rate = 0.01
# Creating model using the Sequential method in tensorflow
def build_model_using_sequential():
    model = Sequential([
        Dense(hidden_units1, kernel_initializer='normal', activation='tanh',  kernel_regularizer='l2'),
        Dropout(0.2),
        Dense(hidden_units2, kernel_initializer='normal', activation='tanh',  kernel_regularizer='l2'),
        Dropout(0.2),
        Dense(hidden_units3, kernel_initializer='normal', activation='tanh'),
        Dense(1, kernel_initializer='normal', activation='relu')
    ])
    return model
# build the model
model = build_model_using_sequential()
msle = MeanSquaredLogarithmicError()
mse = MeanSquaredError()
model.compile(
    loss=mse, 
    optimizer=Adam(learning_rate=learning_rate), 
    metrics=[mse]
)
# train the model
history = model.fit(
    X_train_scaled.values, 
    y_train.values, 
    epochs=60, 
    batch_size=64,
    validation_data=(X_test,y_test),
    callbacks=[callback]
)
X_test['prediction'] = model.predict(X_test_scaled)
results = pd.concat([X_test["prediction"], y_test], axis=1)
results.reset_index(drop=True)
#ouput the RMSE of the model - under 5
((results.prediction - results.AAPL) ** 2).mean() ** .5


# In[17]:


#use the model to predict sample stock prices with some data points
X_test['prediction'] = model.predict(X_test_scaled)
X_test['prediction']


# In[18]:


#the real data points to be compared with predicted values above
y_test


# In[19]:


#put real and predicted values against each other to observe differences
def compare(X_test, y_test):
    results = pd.concat([X_test["prediction"], y_test], axis=1)
    results.reset_index(drop=True)
    results["differences"] = results["prediction"]-results["AAPL"]
    return results
results = compare(X_test, y_test)
results


# In[20]:


#plot the differences on a histogram
plt.hist(results["differences"], density=False, bins=15)  # density=False would make counts
plt.ylabel('Count')
plt.xlabel('Differences');


# In[21]:


#successfully created a model with a RMSE of under 5, reasonably accurate, but room for improvement

