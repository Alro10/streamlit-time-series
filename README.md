# streamlit time series API

This is a basic API made using streamlit to analyze time series.

<p align="center">
<img src="https://github.com/Alro10/streamlit-time-series/blob/master/src/tool.png" alt="alt text" width="90%" height="90%">
</p>

## Quick run

- ```virtualenv venv -p python3.7```

- ```source venv/bin/activate```

- ```pip install -r requirements.txt```

Go to src directory

- ```streamlit run app.py```

## Docker

- ```docker build -t streamlit-ts-ml:0.1.0 -f Dockerfile .```

- ```docker run -p 8501:8501 streamlit-ts-ml:0.1.0```