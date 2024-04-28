# My MLFlow setup

I am using [MLFlow](https://mlflow.org/) to track experiments.
I have set up a virtual machine on [Yandex Compute Cloud](https://yandex.cloud/ru/services/compute) running an MLFlow server behind an [nginx](https://nginx.org/en) reverse proxy for authorization.

## Tracking locally

Locally, I am keeping `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` in a git-excluded `.env` file.

```python
assert dotenv.load_dotenv()
```

## Tracking from Kaggle notebooks

```python
import os

import kaggle_secrets
import mlflow


user_secrets = kaggle_secrets.UserSecretsClient()

os.environ["MLFLOW_TRACKING_URI"] = user_secrets.get_secret("mlflow-uri")
os.environ["MLFLOW_TRACKING_USERNAME"] = user_secrets.get_secret("mlflow-username")
os.environ["MLFLOW_TRACKING_PASSWORD"] = user_secrets.get_secret("mlflow-password")
```

## Tracking from Colab notebooks

```python
import os

import mlflow
from google.colab import userdata


os.environ["MLFLOW_TRACKING_URI"] = userdata.get("mlflow-uri")
os.environ["MLFLOW_TRACKING_USERNAME"] = userdata.get("mlflow-username")
os.environ["MLFLOW_TRACKING_PASSWORD"] = userdata.get("mlflow-password")
```
