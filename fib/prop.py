from binance import Client
import pandas as pd
import numpy as np
import csv
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from prophet import Prophet
import matplotlib.pyplot as plt 
import pandas_ta as ta
import seaborn as sns

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

client=Client(None,None)
searcing=['BTCBUSD','LTCUSDT']

print(searcing)

def getdata(symbol,periods,started,finished):
    formasyon=client.get_historical_klines(symbol,periods,started,finished)
    return formasyon

def createdcsv(coindata,formasyon):
    try:
        CsvFile=open(coindata+'.csv','w',newline='')
        created=csv.writer(CsvFile,delimiter=',')
        for index in formasyon:
            created.writerow(index)
        CsvFile.close()
    except:
        print('Could Not Create File')
def create():
    for coin in searcing:
        createdcsv(coin,getdata(coin,Client.KLINE_INTERVAL_1DAY,'1 January 2018','5 December 2022'))
        print(f'{coin} #Dataset Created...')

readingcsv='BTCBUSD.csv'
title=['DateTime','Open','High','Low','Close','Volume','CloseTime','qab','nat','tbbay','tbgav','ignore']
dataframe=pd.read_csv(readingcsv,names=title)
print(dataframe.tail())

def calculate(timestap):
    return dt.fromtimestamp(timestap/1000)

tarih=dataframe['DateTime']
temp=[]
for value in range(len(tarih)):
    temp.append(calculate(tarih[value]))
dataframe['DateTime']=temp
print(dataframe.head())
dataframe=dataframe[['DateTime','Close']]
df_close=dataframe['Close']
df=dataframe.copy()
print()
df=df.rename({'DateTime':'ds','Close':'y'},axis='columns')
print(df.head())
X=df['ds']
Y=df['y']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
print(Y_train)
print(Y_test)
m=Prophet(yearly_seasonality=True)
m.fit(df)
future = m.make_future_dataframe(periods=14)

future.tail()
forecast = m.predict(future)
m.plot(forecast)
plt.show()

print('MOVING AVAREGE:')
ma5=ta.ma('sma',df['y'],length=5)
ma25=ta.ma('sma',df['y'],length=25)
ma50=ta.ma('sma',df['y'],length=50)
mas=pd.concat([dataframe['DateTime'],dataframe['Close'],ma5,ma25,ma50],axis=1)
mas.columns=['date','close','low','mid','high']
mas.dropna(axis=0,inplace=True)
print(mas.head())

#MA MODEL
moving=list(mas['close'])
model= ARIMA(moving, order=(0, 0, 1))
model_fit = model.fit()
yhat = model_fit.predict(len(moving),len(moving))
print(yhat)

def arımamodels():
    readingcsv='BTCBUSD.csv'
    title=['DateTime','Open','High','Low','Adj Close','Volume','CloseTime','qab','nat','tbbay','tbgav','ignore']
    arimas=pd.read_csv(readingcsv,names=title)

    tarih=arimas['DateTime']
    temp=[]
    for value in range(len(tarih)):
        temp.append(calculate(tarih[value]))
    arimas['DateTime']=temp
    print(arimas.head())
    
    to_row=int(len(df)*0.9)
    print(to_row)
    trading=list(arimas[0:to_row]['Adj Close'])
    test=list(arimas[to_row:]['Adj Close'])

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.plot(arimas[0:to_row]['Adj Close'],'green')
    plt.plot(arimas[to_row:]['Adj Close'],'blue')
    models=[]
    testobser=len(test)
    for i in range(testobser):
        model=ARIMA(trading,order=(4,1,0))
        model_fit=model.fit()

        output=model_fit.forecast()
        yhat=list(output[0])[0]
        models.append(yhat)
        actual=test[i]
        trading.append(actual)
    print(model_fit.summary())

    plt.figure(figsize=(15,9))
    plt.grid(True)
    data_range=arimas[to_row:]['DateTime']
    plt.plot(data_range,models,color='blue',marker='o')
    plt.plot(data_range,test,color='red')
    plt.show()






















