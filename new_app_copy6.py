from tkinter.ttk import Style
from matplotlib import image
import streamlit as st
#from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from pyexpat import model
from keras.models import load_model
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import base64
# Page tab config
st.set_page_config(
    page_title="Home_page",
    page_icon="📈",
)

# ````````````````````````````````````````````````````````````````````
# Menuabr/navbar
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed top navbar-expand-lg navbar-dark", style="background-color: #FF5733;">
  <a class="navbar-brand" href="https://finance.yahoo.com/">Yahoo Finance</a>
 <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="https://www.nseindia.com/">NSE India <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://www.moneycontrol.com/" target="_blank">moneycontrol</a>
      </li>
 </div>
</nav>
""", unsafe_allow_html=True)
# ````````````````````````````````````````````````````````````````````
@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image.jpg")

# ````````````````````````````````````````````````````````````````````
# page background
page_bg_img = f"""
<style>
[data-testid="stSidebar"]>div:first-child{{
background-image:url("data:image/png;base64,{img}");
background-position:center;}}


[data-testid="stAppViewContainer"]{{
background-image:url(https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80);
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
# ````````````````````````````````````````````````````````````````````
# mennu bar
# st.markdown("""
# <nav class="navbar navbar-light bg-light">
#   <span class="navbar-brand mb-0 h1">Deepak</span>
# </nav>""", unsafe_allow_html=True)

# ``````````````````````````````````````````````````````````````````````````````````````````````````````````

# # 3. CSS style definitions
# selected = option_menu(None, ["Home", "Task", "Setting"],
#                        icons=['house', 'list-task', 'gear'],
#                        menu_icon="cast", default_index=0, orientation="horizontal",
#                        styles={
#     "container": {"padding": "0!important", "background-color": "#fafafa"},
#     "icon": {"color": "red", "font-size": "25px"},
#     "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
#     "nav-link-selected": {"background-color": "green"},
# })

# selected


# App Title.
st.title('Stock, Commodities, Crypto Trend Prediction 📈')
# `````````````````````````````````
col1, col2, col3 = st.columns(3)
with col1:
    pass
with col2:
    pass
with col3:
    st.markdown('''##### Made with :heart: by ''')
    st.markdown("""
    - ###### 👩🏻‍💻[Shital hande](https://www.youtube.com/)
    - ###### 🧑🏻‍💻[Ram Dandale](https://www.youtube.com/)
    - ###### 🧑🏻‍💻[Pradip Mali](https://www.youtube.com/)
    - ###### 🧑🏻‍💻[Deepakkumar Mavaskar](https://www.youtube.com/)""")
    

# ```````````````````````````````

# Slidebar design(size) code
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


st.write('---')

st.sidebar.subheader('**👈Query parameter**')
date = st.sidebar.date_input("Enter the start date")  # 16/8/3
start_date = date
date = st.sidebar.date_input("Enter the end date")
end_date = date
text = st.write('#### Date from', start_date, 'to', end_date)

# ticker_list=pd.read_csv('');
user_input = st.sidebar.text_input('Enter Stock Ticker')
# ticker_symbol=st.

try:
    def load_data(user_input):
        data = yf.download(user_input, start_date, end_date)
        data.reset_index(inplace=True)
        return data

    if (user_input != 0):
        df_load_state = st.text("Loading the data...")
        df = load_data(user_input)
        df_load_state.text("Loading the data ... Done!")
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

st.write('---')
# Visualisation
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

st.write('---')

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

st.write('---')

# spliting data into trainging and testing
try:
    if (df.shape[0] != 0):
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(
            df['Close'][int(len(df)*0.70):int(len(df))])
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # load test_model
        model = load_model('keras_model10.h5')

        past_days = data_training.tail(100)
        final_df = past_days.append(data_testing, ignore_index=True)
        input_data = scaler.fit_transform(final_df)
except:
    er = NameError('name \'df\' is not defined')
    st.exception(er)

try:
    x_test = []
    y_test = []
    if (len(input_data) != 0):
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)

    scaler = scaler.scale_

    scale_factor = 1/scaler[0]

    y_predicted = y_predicted*scale_factor
    y_test = y_test*scale_factor


except:
    er = NameError("name 'input_data' is not defined")
    st.exception(er)

# y_predicted1=np.reshape(y_predicted,(994))
# final graph
st.subheader('Predictions vs Origional')
# try:
#     if (len(y_test) != 0 and len(y_predicted != 0)):
y_predicted1 = np.reshape(y_predicted, (220  ,))


def plot_data3():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=y_test, name="Origional Price"))
    fig.add_trace(go.Scatter(
        x=df['Date'], y=y_predicted1, name="Predicted Price"))
    fig.layout.update(
        title_text="Origional Vs Predicted Price Chart", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_data3()
# except:
#     er = ValueError('Input could not be cast to an at-least-1D NumPy array')
#     st.exception(er)
