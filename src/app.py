import streamlit as st
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

st.title('Time Series Analysis')

def tsplot(y, lags=None, figsize=(20, 12), style='bmh'):
    """
        Plot time series, its ACF and PACF
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()

def null_values(df):
    null_test = (df.isnull().sum(axis = 0)/len(df)).sort_values(ascending=False).index
    null_data_test = pd.concat([
        df.isnull().sum(axis = 0),
        (df.isnull().sum(axis = 0)/len(df)).sort_values(ascending=False),
        df.loc[:, df.columns.isin(list(null_test))].dtypes], axis=1)
    null_data_test = null_data_test.rename(columns={0: '# null', 
                                        1: '% null', 
                                        2: 'type'}).sort_values(ascending=False, by = '% null')
    null_data_test = null_data_test[null_data_test["# null"]!=0]
    
    return null_data_test

def main():
    # As an example, let's look at real mobile game data. Specifically, we will look into ads watched per hour and in-game currency spend per day:
    # ads = pd.read_csv('ads.csv', index_col=['Time'], parse_dates=['Time'])
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    # Add a slider to the sidebar:
    x = st.sidebar.slider(
        'Select a lang',
        50, 60
    )
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Exploratory Data Analysis")
        # Show dataset
        if st.checkbox("Show Dataset"):
            rows = st.number_input("Number of rows", 5, len(data))
            st.dataframe(data.head(rows))
        # Show columns
        if st.checkbox("Columns"):
            st.write(data.columns)
        # Show Shape
        if st.checkbox("Shape of Dataset"):
            data_dim = st.radio("Show by", ("Rows", "Columns", "Shape"))
            if data_dim == "Columns":
                st.text("Number of Columns: ")
                st.write(data.shape[1])
            elif data_dim == "Rows":
                st.text("Number of Rows: ")
                st.write(data.shape[0])
            else:
                st.write(data.shape)
        # Check null values in dataset
        if st.checkbox("Check null values"):
            nvalues = null_values(data)
            st.write(nvalues)
        # Plot time series, ACF and PACF
        if st.checkbox("Select column as time series"):
            columns = data.columns.tolist()
            selected = st.multiselect("Choose", columns)
            series = data[selected]
            if st.button('Plot Time Series, ACF and PACF'):
                tsplot(series, lags=x)
                st.pyplot()



if __name__ == "__main__":
    main()