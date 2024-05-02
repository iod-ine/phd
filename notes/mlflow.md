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

---

### Server setup

- Fresh VM setup: `apt update && apt install git nginx tmux`
- Libraries for Python install: `apt install python3-pip libssl-dev libncurses-dev libsqlite3-dev libbz2-dev libreadline6-dev libffi-dev liblzma-dev`
- Install `pyenv`: `curl https://pyenv.run | bash`
- Install Python: `pyenv install 3.10.13`, `pyenv global 3.10.13`
- Install MLflow: `pip install mlflow`
- Start MLflow: `mlflow server --backend-store-uri sqlite:///mlflow.db`
- Setup password: `echo "username:$(openssl passwd -apr1)" >> /etc/nginx/.htpasswd`
- Setup site: `vim /etc/nginx/sites-available/mlflow` (config below)
- Enable site: `ln -s /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/`
- Restart service: `systemctl restart nginx`

#### `nginx` configuration

```nginx
server {
 listen 8080;
 server_name 123.456.789.10;
 client_max_body_size 20M;  # Increase to allow logging model checkpoints

 location / {
     auth_basic "Protected Area";
     auth_basic_user_file /etc/nginx/.htpasswd;  # Path to the password file
     proxy_pass http://127.0.0.1:5000;
 }
}
```
