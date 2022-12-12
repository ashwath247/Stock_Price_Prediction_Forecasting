import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from nsepy import get_history
from datetime import date
import datetime
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

def forecasting(last_30_days_scaled, last_30_days, model, scaler):
    x=[]
    for i in last_30_days_scaled.tolist():
        x.append(i[0])
    lst_output=[]
    n_steps=30
    i=0
    while(i<30):
        if(len(x)>30):
            last_30_days_scaled=np.array(x[1:])
            last_30_days_scaled=last_30_days_scaled.reshape(1,-1)
            last_30_days_scaled = last_30_days_scaled.reshape((1, n_steps, 1))

            yhat = model.predict(last_30_days_scaled, verbose=0)
            x.extend(yhat[0].tolist())
            x=x[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            last_30_days_scaled = last_30_days_scaled.reshape((1, n_steps,1))
            yhat = model.predict(last_30_days_scaled, verbose=0)
            x.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    day_new=np.arange(1,31)
    day_pred=np.arange(31,61)
    pred_vals=scaler.inverse_transform(lst_output)
            
    values=pd.DataFrame({'index':day_new, 'value': last_30_days.tolist(),'label':['last 30 days' for i in range(len(day_new))]})
    predictions=pd.DataFrame({'index': day_pred, 'value': pred_vals.tolist(),'label':['predictions' for i in range(len(day_pred))]})
    predictions['value']=predictions['value'].apply(lambda x: x[0])
    values['value']=values['value'].apply(lambda x: x[0]) 
    final=pd.concat([values, predictions], axis=0)
    return final
    


def main():
    st.title("NSE real time stock analysis, prediction and forecasting")
    
    st.header("Select the stock and check its next day predicted value")
    
    
    choose_stock = st.sidebar.selectbox("Choose the Stock!",["NONE","Reliance","TCS","Infosys","HDFC"])
    
    if(choose_stock == "Reliance"):
        df1 = get_history(symbol='reliance', start=date(2010,1,1), end=date.today())
        df1['Date'] = df1.index
        
        st.header("Reliance NSE Last 5 Days DataFrame:")
        
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df1.tail())
        
        new_df = df1.filter(['Close'])
        
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        
        last_30_days = new_df[-30:].values
     
        last_30_days_scaled = scaler.transform(last_30_days)
        X_test = []

        X_test.append(last_30_days_scaled)
       
        X_test = np.array(X_test)
        
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        model = load_model("reliance.h5")
        
        pred_price = model.predict(X_test)
       
        pred_price = scaler.inverse_transform(pred_price)
       
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price[0][0])
        
        if st.checkbox('Show next 30 days forecasting:'):
            final=forecasting(last_30_days_scaled, last_30_days, model, scaler)
            
            fig=px.line(final, x='index',y='value', color='label')
            st.write(fig)
            
            
            

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df1[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High','Low']])

                
    elif(choose_stock == "TCS"):
        df1 = get_history(symbol='tcs', start=date(2010,1,1), end=date.today())
        df1['Date'] = df1.index
        
        st.header("TCS NSE Last 5 Days DataFrame:")
        
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df1.tail())
        
        new_df = df1.filter(['Close'])
        
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        
        last_30_days = new_df[-30:].values
     
        last_30_days_scaled = scaler.transform(last_30_days)
        X_test = []

        X_test.append(last_30_days_scaled)
       
        X_test = np.array(X_test)
        
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        model = load_model("TCS.h5")
        
        pred_price = model.predict(X_test)
       
        pred_price = scaler.inverse_transform(pred_price)
       
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price[0][0])
        
        if st.checkbox('Show next 30 days forecasting:'):
            final=forecasting(last_30_days_scaled, last_30_days, model, scaler)
            
            fig=px.line(final, x='index',y='value', color='label')
            st.write(fig)

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df1[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High','Low']])

    elif(choose_stock == "Infosys"):
        df1 = get_history(symbol='infy', start=date(2010,1,1), end=date.today())
        df1['Date'] = df1.index
        
        st.header("Infosys NSE Last 5 Days DataFrame:")
        
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df1.tail())
        
        new_df = df1.filter(['Close'])
        
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        
        last_30_days = new_df[-30:].values
     
        last_30_days_scaled = scaler.transform(last_30_days)
        X_test = []

        X_test.append(last_30_days_scaled)
       
        X_test = np.array(X_test)
        
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        model = load_model("infy.h5")
        
        pred_price = model.predict(X_test)
       
        pred_price = scaler.inverse_transform(pred_price)
       
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price[0][0])
        
        if st.checkbox('Show next 30 days forecasting:'):
            final=forecasting(last_30_days_scaled, last_30_days, model, scaler)
            
            fig=px.line(final, x='index',y='value', color='label')
            st.write(fig)

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df1[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High','Low']])
        
        
    if(choose_stock == "HDFC"):
        df1 = get_history(symbol='hdfc', start=date(2010,1,1), end=date.today())
        df1['Date'] = df1.index
        
        st.header("HDFC NSE Last 5 Days DataFrame:")
        
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df1.tail())
        
        new_df = df1.filter(['Close'])
        
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        last_30_days = new_df[-30:].values
        last_30_days_scaled = scaler.transform(last_30_days)
        X_test = []
        X_test.append(last_30_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        model = load_model("HDFC.h5")
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price[0][0])
        
        if st.checkbox('Show next 30 days forecasting:'):
            final=forecasting(last_30_days_scaled, last_30_days, model, scaler)
            
            fig=px.line(final, x='index',y='value', color='label')
            st.write(fig)

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df1[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High','Low']])

if __name__ == '__main__':

    main()
        
