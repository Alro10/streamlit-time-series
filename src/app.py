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

@st.cache
def tsplot(y, lags=None, figsize=(20, 12), style='bmh'):
    """
        Plot time series, its ACF and PACF
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
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

def main():
    # As an example, let's look at real mobile game data. Specifically, we will look into ads watched per hour and in-game currency spend per day:
    ads = pd.read_csv('ads.csv', index_col=['Time'], parse_dates=['Time'])
    # Add a slider to the sidebar:
    x = st.sidebar.slider(
        'Select a lang',
        50, 60
    )
    c = tsplot(ads.Ads, lags=x)
    st.write(ads)
    st.pyplot()

if __name__ == "__main__":
    main()