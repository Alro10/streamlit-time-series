FROM python:3.7

RUN apt-get update && apt-get upgrade -y

WORKDIR /usr/src/app/streamlit-time-series

# Exposing default port for streamlit
EXPOSE 8501

# Install requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy necessary files
COPY . .

# Launch app when container is run
CMD streamlit run src/app.py