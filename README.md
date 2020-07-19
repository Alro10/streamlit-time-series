# streamlit time series API

This is a basic API made using streamlit to analyze time series. You can see the deployment, hosted on IBM cloud - Kubernetes Cluster, here: **http://159.122.181.167:30498/** 

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

Then stop the process with the following command.

```shell
$ docker kill <weird_name>
<weird_name>
$
```

## IBM Cloud - Deploy

- If you need help for deploying on Kubernetes:

- For tag and push (docker image) to container registry:
  - ```docker tag streamlit-ts-ml:0.2.0 us.icr.io/ml-api/streamlit-ts-ml:0.2.0```

  - ```docker push us.icr.io/ml-api/streamlit-ts-ml:0.2.0```

Go to deploy directory and run the following commands:

- For deployment:
  - ```kubectl apply -f deployment.yaml```

- For service:
  - ```kubectl apply -f service.yaml```

  Do not feel angry if the first try does not work, kubernetes and cloud is not too easy...but the result is really awesome!