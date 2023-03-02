import os
from tkinter import *
from tkinter.ttk import Style
from matplotlib import image
from pandas_profiling import ProfileReport
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from pyexpat import model
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import base64


# 1. Page tab config

st.set_page_config(
    page_title="Home_page",
    page_icon="📈",
)


# 2. Menuabar/Navigation bar

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed top navbar-expand-lg navbar-dark", style="background-color: #FF5733;">
  <a class="navbar-brand" href="https://finance.yahoo.com/">Yahoo Finance</a>
 <div class="navbar fixed top navbar-expand-lg navbar-dark" id="navbarNav">
    <ul class="navbar-nav">     
 </div>
</nav>
""", unsafe_allow_html=True)


# 3. Program to add image
@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image.jpg")


#4.  sidebar background/Page Background

page_bg_img = f"""
<style>
[data-testid="stSidebar"]>div:first-child{{
background-image:url("data:image/png;base64,{img}");
background-position:center;}}

[data-testid="stAppViewContainer"]{{
background-image:url(https://c1.wallpaperflare.com/path/787/792/907/abstract-art-abstract-art-painting-2561f1d754ccf645b4cbf9602bafbf43.jpg);
background-size: cover;}}

[data-testid="stHeader"]{{
    background-color:rgba(0,0,0,0);
}}

[data-testid="st.Toolbar]{{
right: 2rem;
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# 5. App Title.
st.title('📈Stock Commodity Crypto Prediction')


# 6. Slidebar design(size) code
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 7. sidebar program

st.sidebar.subheader('**👈Query parameter**')
date = st.sidebar.date_input("Enter the start date")
start_date = date
date = st.sidebar.date_input("Enter the end date")
end_date = date
text = st.write('#### Date from', start_date, 'to', end_date)

user_input = st.sidebar.text_input('Enter Stock Ticker')
ticker_symbol = st.write("#### Stock Ticker  =  ", user_input)

try:
    def load_data(user_input):
        data = yf.download(user_input, start_date, end_date)
        data.reset_index(inplace=True)
        return data

    if (user_input != 0):
        df_load_state = st.write("#### Loading the data...")
        df = load_data(user_input)
        df_load_state = st.write("#### Data Loading  -----------> Done!")
        # tickerData=yf.Ticker(tickerS)
        st.subheader("Data")
        st.write(df.tail())
        st.subheader("Summery data")
        st.write(df.describe())

except:
    e = RuntimeError(
        'This is exception of type RuntimeError\nTicker Not Provided')
    print('Ticker Not Provided')
    st.exception(e)

# 8. pandas profiling 

# try:
#     if (1 == 1):
#         directory = "./"

#         files_in_directory = os.listdir(directory)
#         filtered_files = [
#             file for file in files_in_directory if file.endswith(".html")]
#         for file in filtered_files:
#             path_to_file = os.path.join(directory, file)
#             os.remove(path_to_file)

#         prof = ProfileReport(df)
#         prof.to_file(output_file=user_input+".html")
#     else:
#         pass
# except:
#     e = FileExistsError("File already exist error")
#     st.exception(e)

# 9. Closing price line chart with slider(first graph)

st.subheader('Closing price VS Time Chart')
try:
    if (len(df) != 0):
        def plot_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['Close'], name='stock_close'))
            fig.layout.update(title_text="Time series data",
                              xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_data()
    else:
        st.write("dataframe is not generated to plot the graph")

except:
    e = ModuleNotFoundError("module is not found")
    st.exception(e)


# 10. Closing price VS 100 DMA and 200 DMA line chart with slider(second graph)
 
st.subheader('Closing Price VS Time Chart with 100MA & 200MA')

try:
    if (len(df) != 0):
        def plot_data2():
            ma1 = df.Close.rolling(100).mean()
            ma2 = df.Close.rolling(200).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=ma1, name='100 Day MA'))
            fig.add_trace(go.Scatter(x=df['Date'], y=ma2, name='200 Day MA'))
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['Close'], name='Origional'))
            fig.layout.update(
                title_text="100 Day and 200 Day moving averages chart", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_data2()
    else:
        st.write("dataframe is not generated to plot the graph")
except:
    e = ModuleNotFoundError("module is not found")
    st.exception(e)


# 11. spliting data into trainging and testing
try:
    if (df.shape[0] != 0):
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(
            df['Close'][int(len(df)*0.70):int(len(df))])

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

# 12. Creating x_train y_train dataset after spliting data into training and testing
     
        x_train = []
        y_train = []

        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i-100:i])
            y_train.append(data_training_array[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        past_days = data_training.tail(100)
        final_df = past_days.append(data_testing, ignore_index=True)
        input_data = scaler.fit_transform(final_df)
except:
    er = NameError('name \'df\' is not defined')
    st.exception(er)

# 13. Creating x_test y_test dataset after spliting data into training and testing

try:
    x_test = []
    y_test = []
    if (len(input_data) != 0):
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # 14 loading LSTM model
    
    model = load_model('keras_model100.h5')

    # 15. Making predictions on test data  checking the RMSE Performance Matrix
    
    test_predict = model.predict(x_test)

    # 16 . Get the root mean squared error of Test data for LSTM model of 100 epochs (RMSE)
    
    st.subheader("Model Prediction")
    st.markdown("""- ##### Thumb Rule in Regression Analysis""")
    st.write("It can be said that RMSE values between 0.2 and 0.5 shows that the model can relatively predict the data accurately. In addition, Adjusted R-squared more than 0.75 is a very good value for showing the accuracy. In some cases, Adjusted R-squared of 0.4 or more is acceptable as well.")
    
    # rmse score value
    rmse = np.sqrt(np.mean(((test_predict - y_test)**2)))
    st.write("rmse = ", rmse)
    # accuracy score


    data_test = y_test
    
    # 17. Transform back to get original values
    
    test_predict = scaler.inverse_transform(test_predict)
    y_test = y_test.reshape(-1, 1)
    y_test = scaler.inverse_transform(y_test)


except:
    er = NameError("name 'input_data' is not defined")
    st.exception(er)

st.write('---')

# 18. final graph

st.subheader('Predictions vs Origional')

def flat(lis):
    flatList = []
    # Iterate with outer list
    for element in lis:
        if type(element) is list:
            # Check if type is list than iterate through the sublist
            for item in element:
                flatList.append(item)
        else:
            flatList.append(element)
    return flatList


try:
    if (len(y_test) != 0 and len(test_predict != 0)):
        y1 = y_test.tolist()
        y2 = test_predict.tolist()
        y_test = flat(y1)
        test_predict = flat(y2)

    def plot_data3():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=None, y=y_test, name="Origional Price"))
        fig.add_trace(go.Scatter(
            x=None, y=test_predict, name="Predicted Price"))
        fig.layout.update(
            title_text="Origional Vs Predicted Price Chart", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_data3()
except:
    er = ValueError('Input could not be cast to an at-least-1D NumPy array')
    st.exception(er)


# 19. getting last 100 days record

try:
    data_test = pd.DataFrame(data_test, columns=['Data_test'])
    x_input = data_test.tail(100).values.reshape(1, -1)
    # Creating the list of last 100 data
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

# 20. Predicting next 30 days price suing the current data

    lst_output = []
    n_steps = 100
    i = 0
    while (i < 30):

        if (len(temp_input) > 100):
            # print(temp_input)
            x_input = np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            # print(x_input)
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            # print(temp_input)
            lst_output.extend(yhat.tolist())
            i = i+1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i+1

except:
    er = NameError("name 'input_data' is not defined")
    st.exception(er)

# 21. Creating a dummy plane to plot graph one after another
df1 = df['Close']
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

st.subheader('prediction of next 30 days price on basis of last 100 day cloing price')

day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)

fig1 = plt.figure(figsize=(12, 6))
plt.ylabel("Price")
plt.xlabel("Time")
plt.plot(day_new, scaler.inverse_transform((df1[(len(df1)-100):])))
plt.plot(day_pred, scaler.inverse_transform(lst_output))
st.pyplot(fig1)

df2 = df1.tolist()
df2.extend(lst_output)

# Creating final data for plotting
final_graph = scaler.inverse_transform(df2).tolist()

# 22. Plotting final results with predicted value after 30 Days
st.subheader('merging of next 30 day predicted value with  all data value')

fig3 = plt.figure(figsize=(12, 6))
plt.plot(final_graph,)
plt.ylabel("Price")
plt.xlabel("Time")
plt.title("{0} prediction of next 30 Day price in INR".format('stock_symbol'))
plt.axhline(y=final_graph[len(final_graph)-1], color='red', linestyle=':',
            label='NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]), 2)))
plt.legend()
st.pyplot(fig3)

# 24. Project submitted by student name
col1, col2, col3 = st.columns(3)
with col1:
    pass
with col2:
    pass
with col3:
    st.markdown('''#### Made with :heart: by ''')
    st.markdown("""
    - ##### 👩🏻‍💻[Shital hande](https://www.youtube.com/)
    - ##### 🧑🏻‍💻[Ram Dandale](https://www.youtube.com/)
    - ##### 🧑🏻‍💻[Pradip Mali](https://www.youtube.com/)
    - ##### 🧑🏻‍💻[Deepak Mavaskar](https://www.youtube.com/)""")
