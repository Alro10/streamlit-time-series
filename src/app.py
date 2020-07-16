import streamlit as st
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#sktime models
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.plotting.forecasting import plot_ys
from sktime.forecasting.arima import AutoARIMA

import warnings
warnings.filterwarnings('ignore')

st.title('Time Series - ML')

@st.cache
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

def types(df):
    return pd.DataFrame(df.dtypes, columns=['Type'])

def forecasting_autoarima(y_train, y_test, s):
    fh = np.arange(len(y_test)) + 1
    forecaster = AutoARIMA(sp=s)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    st.pyplot()

def main():
    st.sidebar.title("What to do")
    activities = ["Exploratory Data Analysis", "Plotting and Visualization", "Building Model", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    # Upload file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None and choice == "Exploratory Data Analysis":
        data = pd.read_csv(uploaded_file)
        st.subheader(choice)
        # Add a slider to the sidebar:
        st.sidebar.markdown("# Lang")
        x = st.sidebar.slider(
            'Select a lang',
            50, 60
        )
        # Show dataset
        if st.checkbox("Show Dataset"):
            rows = st.number_input("Number of rows", 5, len(data))
            st.dataframe(data.head(rows))
        # Show columns
        if st.checkbox("Columns"):
            st.write(data.columns)
        # Data types
        if st.checkbox("Column types"):
            st.write(types(data))
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
        # Show Data summary
        if st.checkbox("Show Data Summary"):
            st.text("Datatypes Summary")
            st.write(data.describe())
        # Plot time series, ACF and PACF
        if st.checkbox("Select column as time series"):
            columns = data.columns.tolist()
            selected = st.multiselect("Choose", columns)
            series = data[selected]
            if st.button('Plot Time Series, ACF and PACF'):
                tsplot(series, lags=x)
                st.pyplot()

    elif uploaded_file is not None and choice == "Plotting and Visualization":
        st.subheader(choice)
        data = pd.read_csv(uploaded_file)
        df = data.copy()
        all_columns = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot", ["area", "line", "scatter", "pie", "bar", "correlation"]) 
        
        if type_of_plot=="line":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.line_chart(cust_data)
        
        elif type_of_plot=="area":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.area_chart(cust_data)  
        
        elif type_of_plot=="bar":
            select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
            cust_data = df[select_columns_to_plot]
            st.bar_chart(cust_data)
        
        elif type_of_plot=="pie":
            select_columns_to_plot = st.selectbox("Select a column", all_columns)
            st.write(df[select_columns_to_plot].value_counts().plot.pie())
            st.pyplot()
        
        elif type_of_plot=="correlation":
            st.write(sns.heatmap(df.corr(), annot=True, linewidths=.5, annot_kws={"size": 7}))
            st.pyplot()

        elif type_of_plot=="scatter":
            st.write("Scatter Plot")
            scatter_x = st.selectbox("Select a column for X Axis", all_columns)
            scatter_y = st.selectbox("Select a column for Y Axis", all_columns)
            st.write(sns.scatterplot(x=scatter_x, y=scatter_y, data = df))
            st.pyplot()

    elif choice == "Building Model":
        st.subheader(choice)
         # Add a slider to the sidebar:
        st.sidebar.markdown("# Seasonal")
        s = st.sidebar.slider(
            'Select a s',
            24, 48
        )
        data = pd.read_csv(uploaded_file)
        df = data.copy()
        st.write("Select the columns to use for training")
        columns = df.columns.tolist()
        selected_column = st.multiselect("Select Columns", columns)
        new_df = df[selected_column]
        st.write(new_df)

        if st.button("Train/Test Split"):
            y_train, y_test = temporal_train_test_split(new_df.T.iloc[0])
            st.text("Train Shape")
            st.write(y_train.shape)
            st.text("Test Shape")
            st.write(y_test.shape)
            plot_ys(y_train, y_test, labels=["y_train", "y_test"])
            st.pyplot()
            
        if st.button("Training a Model"):
            y_train, y_test = temporal_train_test_split(new_df.T.iloc[0])
            model_selection = st.selectbox("Model to train", ["AutoArima", "LSTM", "MLP", "RNN"])
            if model_selection == "AutoArima":
                forecasting_autoarima(y_train, y_test, s)

if __name__ == "__main__":
    main()
    