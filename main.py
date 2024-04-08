from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from keras.models import load_model
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
 
# Declaring our FastAPI instance
app = FastAPI()

class request_body(BaseModel):
    prediction:float


@app.get('/')
def main():
    return {'message': 'Welcome to GeeksforGeeks!'}
 
# Defining path operation for root endpoint
@app.get('/predict')
def predict():
    #with open('Model.pkl',"rb") as file:
    clf=load_model(Model1.pkl)

    scaler=MinMaxScaler(feature_range=(0,1))
    timeinterval=24
    prediction=1
    predict=[]
    exchange=[]

    i=0
    testapi='https://api.twelvedata.com/time_series?symbol=BTC/INR&interval=5min&outputsize=273&timezone=Asia/Kolkata&order=ASC&apikey=e76157c75c3a42649e168c5c206e88ca'
    testdata=requests.get(testapi).json()
    testdatafinal=pd.DataFrame(testdata['values'])

    bitcoinprice=pd.to_numeric(testdatafinal['close'],errors='coerce').values
    testinputs=testdatafinal['close'].values
    testinputs=testinputs.reshape(-1,1)
    modelinputs=scaler.fit_transform(testinputs)

    x_test=[]
    for x in range(timeinterval,len(modelinputs)):
        x_test.append(modelinputs[x-timeinterval:x,0])

    x_test=np.array(x_test)
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    prediction_price=clf.predict(x_test)
    prediction_price=scaler.inverse_transform(prediction_price)

    exchangeapi='https://api.twelvedata.com/exchange_rate?symbol=BTC/INR&timezone=Asia/Kolkata&apikey=e76157c75c3a42649e168c5c206e88ca'
    exchangedata=requests.get(exchangeapi).json()
    #print(exchangedata)

    exchangefinal=exchangedata['rate']
    exchange.append(exchangefinal)


    lastdata=modelinputs[len(modelinputs)+1-timeinterval:len(modelinputs)+1,0]
    lastdata=np.array(lastdata)
    lastdata=np.reshape(lastdata,(1,lastdata.shape[0],1))
    prediction=clf.predict(lastdata)
    prediction=scaler.inverse_transform(prediction)
    predict.append(prediction[0][0])
    pre=prediction[0][0]
    print(pre)
    
    return {'message': np.float64(pre) }
 
# Defining path operation for /name endpoint
"""@app.get('/{name}')
def hello_name(name : str): 
    # Defining a function that takes only string as input and output the
    # following message. 
    return {'message': f'Welcome to GeeksforGeeks!, {name}'}"""
