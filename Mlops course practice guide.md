```markdown
# MLOps Course

## Hands-On Practice Guide

Section-by-Section Practice Tasks | Real Commands | What to Build After Each Lecture

*12 Sections | 69 Lectures | Section-Mapped Hands-On Exercises*

---

### How to Use This Practice Guide

This guide maps directly to your 12-section course. After you watch each section, come here and complete the practice tasks for that section. Do not move forward until you have passed the 'Done When' check for every task.

---

**3 Rules to Follow**

1. **Watch the lectures FIRST, then practice.** Do not code while watching.
2. **Every task has a 'Done When' check.** Only move on when that passes.
3. **Commit everything to Git.** Real MLOps means version-controlling all work.

---

### Course Overview

| Course Section | Tool / Platform | Practice Outcome |
|----------------|-----------------|------------------|
| Sections 1-2: Foundations | Git, Python, venv | Project scaffold on GitHub, model training script working |
| Section 3: Data Versioning | DVC + AWS S3 | Dataset versioned, pipeline reproducible with `dvc repro` |
| Section 4: Experiment Tracking | MLflow + PostgreSQL + K8s | Full tracking server running, all runs logged in MLflow UI |
| Section 5: Model Serving | FastAPI, gunicorn, EC2 | Model live on EC2, API returns predictions via public IP |
| Section 6: Docker + Kubernetes | Docker, kubectl, Helm | Containerized model running in K8s, accessible via Ingress |
| Section 7: KServe | KServe + Istio | Model served via KServe InferenceService with V2 protocol |
| Section 8: SageMaker | AWS SageMaker SDK | Training job complete, real-time endpoint tested and deleted |
| Section 9: Kubeflow Pipelines | Kubeflow, KFP SDK v2 | 3-component pipeline running successfully in KFP UI |
| Section 10: CI/CD | GitHub Actions + Argo CD | Full GitOps: push code → CI → manifest update → deploy |

---

## SECTIONS 1 & 2 — Foundations + Project Setup

### Sections 1 & 2: Foundations Practice

These sections cover theory: ML lifecycle, roles, and how MLOps helps. Your practice is to set up the project environment correctly so every later section works smoothly.

**What You Should Understand After These Sections**
- The difference between Data Scientist, ML Engineer, and MLOps Engineer
- Why MLOps exists and what problems it solves for data scientists in production
- The full ML lifecycle: data → experiment → deploy → monitor → retrain
- Where the course project (Wine / Intent Classifier) fits in the lifecycle

---

#### PRACTICE TASK 1.1: Project Scaffold on GitHub

**What to Build:** Create a well-structured MLOps project folder and push it to a new GitHub repo.

**How:** Create folders: `data/`, `src/`, `models/`, `tests/`, `notebooks/`, `.github/workflows/`. Add `README.md` and `.gitignore`. Push to GitHub.

**Done When:** You can run `git log` and see your first commit. Your repo is visible on GitHub.

```bash
# Run these commands one by one in your terminal

mkdir mlops-project && cd mlops-project

git init

mkdir -p data/raw data/processed src models tests notebooks .github/workflows

touch README.md src/__init__.py src/train.py src/evaluate.py

# Create .gitignore file
echo "*.pkl" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".env" >> .gitignore
echo "mlops-env/" >> .gitignore
echo "mlruns/" >> .gitignore

# Create virtual environment
python -m venv mlops-env
source mlops-env/bin/activate  # Windows: mlops-env\Scripts\activate

pip install scikit-learn pandas numpy mlflow dvc fastapi uvicorn pytest joblib boto3

pip freeze > requirements.txt

git add .
git commit -m "feat: initial project scaffold"

git remote add origin https://github.com/YOUR_USERNAME/mlops-project.git
git push -u origin main
```

---

#### PRACTICE TASK 1.2: Write Your First Model (Wine Classifier)

**What to Build:** Write a clean, well-commented `train.py` that trains on the Wine dataset and saves a `.pkl` model.

**How:** Paste the code below into `src/train.py`. Run it. Confirm model appears in `models/` folder.

**Done When:** Running `python src/train.py` prints accuracy >= 0.90 and `models/wine_model.pkl` exists.

```python
# src/train.py

import pandas as pd
import joblib
import os
import json
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train(n_estimators=100, max_depth=5, random_state=42):
    # Load data
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Save raw data
    df.to_csv('data/raw/wine.csv', index=False)
    print('Dataset saved to data/raw/wine.csv')
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train-test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth
    )
    model.fit(X_tr, y_tr)
    
    # Evaluate
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f'Accuracy: {acc:.4f}')
    
    # Save model and metrics
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/wine_model.pkl')
    
    with open('models/metrics.json', 'w') as f:
        json.dump({'accuracy': round(acc,4), 'n_estimators': n_estimators}, f)
    
    print('Model saved: models/wine_model.pkl')

if __name__ == '__main__':
    train()
```

---

## SECTION 3 — Data Versioning with DVC + AWS S3

### Section 3: DVC Practice — Version Your Data

The course covers DVC + AWS S3 as remote storage. Your job: track `wine.csv` with DVC, store it in S3, and create a reproducible pipeline.

**Concepts to Understand Before Practicing**
- Why Git cannot version large data files (blob size limits, binary diffs)
- How DVC stores only a tiny `.dvc` pointer file in Git; data goes to S3 remote
- DVC stages: `dvc.yaml` defines a pipeline with inputs, outputs, and commands
- `dvc repro` runs the full pipeline, only re-running stages that changed

---

#### PRACTICE TASK 3.1: DVC Init + Track Dataset on S3

**What to Build:** Initialize DVC in your project and track `wine.csv` — store it in an AWS S3 bucket.

**How:** Run the DVC setup commands. Create a free S3 bucket in AWS Console. Configure DVC remote. Push data.

**Done When:** `dvc push` completes with no errors. S3 bucket shows the file. `git log` shows `.dvc` commit.

```bash
pip install dvc dvc-s3

dvc init

git add .dvc .dvcignore
git commit -m "feat: initialize DVC"

# Track the dataset file
dvc add data/raw/wine.csv
git add data/raw/wine.csv.dvc data/raw/.gitignore
git commit -m "data: track wine dataset with DVC"

# Configure S3 remote (create bucket in AWS Console first)
dvc remote add -d myremote s3://YOUR-BUCKET-NAME/dvc-store
dvc remote modify myremote region us-east-1

git add .dvc/config
git commit -m "config: add S3 DVC remote"

# Push data to S3
dvc push

# Test: delete local file and pull back from S3
rm data/raw/wine.csv
dvc pull  # Should restore wine.csv from S3
```

---

#### PRACTICE TASK 3.2: Create a Reproducible DVC Pipeline

**What to Build:** Define a DVC pipeline with two stages: `prepare_data` and `train_model`.

**How:** Create `dvc.yaml` with two stages. Run `dvc repro`. Commit the `dvc.yaml` and `dvc.lock` files.

**Done When:** `dvc repro` runs without errors. `dvc dag` shows a two-stage pipeline graph. `dvc.lock` is committed.

```yaml
# dvc.yaml — create this file in project root

stages:
  prepare_data:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
    outs:
      - data/raw/wine.csv
  
  train_model:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/raw/wine.csv
    outs:
      - models/wine_model.pkl
    metrics:
      - models/metrics.json:
          cache: false
```

```bash
# Run the pipeline
dvc repro

# View pipeline graph
dvc dag

# Compare metrics between git commits
dvc metrics show
dvc metrics diff HEAD~1

git add dvc.yaml dvc.lock models/metrics.json
git commit -m "pipeline: add DVC two-stage pipeline"
```

---

## SECTION 4 — Experiment Tracking with MLflow

### Section 4: MLflow Practice — Track Every Experiment

This is the most important section to practice. The course shows MLflow on Kubernetes with PostgreSQL. Start local, then move to the full K8s + PostgreSQL setup from the lecture.

**Practice in 3 Stages**
- **Stage A:** Local MLflow (just to understand the API and UI)
- **Stage B:** MLflow on Kubernetes with SQLite (non-production, good for practice)
- **Stage C:** MLflow on Kubernetes with PostgreSQL (production — matches the course lecture exactly)

---

#### PRACTICE TASK 4.1: Local MLflow — Track 5 Wine Model Experiments

**What to Build:** Add MLflow tracking to `train.py`. Run 5 experiments with different hyperparameters. Compare in MLflow UI.

**How:** Modify `src/train.py` to use `mlflow.start_run()`. Log params and metrics. Run 5 times changing `n_estimators`.

**Done When:** `mlflow ui` shows 5 runs under 'wine-experiment'. Can compare all runs. Best accuracy run identified.

```python
# src/train_mlflow.py — MLflow-tracked version

import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import os
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train(n_estimators=100, max_depth=5):
    mlflow.set_experiment('wine-experiment')
    run_name = 'rf-n' + str(n_estimators) + '-d' + str(max_depth)
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
        mlflow.set_tag('model_type', 'RandomForest')
        
        # Load and split data
        data = load_wine()
        X_tr, X_te, y_tr, y_te = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth
        )
        model.fit(X_tr, y_tr)
        
        # Evaluate
        preds = model.predict(X_te)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average='macro')
        
        # Log metrics
        mlflow.log_metrics({
            'accuracy': acc,
            'f1_macro': f1
        })
        
        # Log model
        mlflow.sklearn.log_model(model, 'model', registered_model_name='WineClassifier')
        
        print('Run logged | acc=' + str(round(acc, 4)))

if __name__ == '__main__':
    for n in [50, 100, 150, 200, 250]:
        train(n_estimators=n, max_depth=5)
```

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Open browser at: http://localhost:5000
# Click wine-experiment -> select all 5 runs -> Click Compare
```

---

#### PRACTICE TASK 4.2: MLflow on Kubernetes with PostgreSQL

**What to Build:** Deploy the MLflow server on a local Kubernetes cluster backed by PostgreSQL (the course production setup).

**How:** Use Kind or Minikube. Deploy PostgreSQL as K8s StatefulSet. Deploy MLflow server Deployment. Port-forward.

**Done When:** `kubectl get pods` shows postgres and mlflow pods Running. MLflow UI accessible. Runs persist after pod restart.

```bash
# Step 1: Create local cluster (install Kind from kind.sigs.k8s.io)
kind create cluster --name mlops-cluster

# Step 2: Create namespace
kubectl create namespace mlflow

# Step 3: Create PostgreSQL secret
kubectl create secret generic postgres-secret -n mlflow \
  --from-literal=POSTGRES_USER=mlflow \
  --from-literal=POSTGRES_PASSWORD=mlflow123 \
  --from-literal=POSTGRES_DB=mlflowdb

# Step 4: Apply postgres StatefulSet manifest
# (create k8s/postgres.yaml with postgres:14 image, ClusterIP service on port 5432)
kubectl apply -f k8s/postgres.yaml -n mlflow

# Step 5: Apply MLflow Deployment manifest
# image: python:3.11-slim
# command: mlflow server
# --backend-store-uri postgresql://mlflow:mlflow123@postgres-svc:5432/mlflowdb
# --default-artifact-root s3://YOUR-BUCKET/mlflow
# --host 0.0.0.0 --port 5000
kubectl apply -f k8s/mlflow.yaml -n mlflow

# Step 6: Port-forward to access UI
kubectl port-forward svc/mlflow-svc 5000:5000 -n mlflow &

# Step 7: Point your code to K8s MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/train_mlflow.py  # Runs now tracked in K8s MLflow with PostgreSQL
```

---

## SECTION 5 — Model Serving on EC2 (FastAPI + WSGI)

### Section 5: Model Serving Practice — Deploy to EC2

The course uses an Intent Classifier. Practice with your Wine model — the architecture is identical. Focus on the deployment pattern, not the specific model.

**Architecture You Are Building**
- Client → Internet → EC2 Security Group → gunicorn (WSGI) → FastAPI app → Model (.pkl)
- The course covers WSGI in detail — gunicorn is the WSGI server, FastAPI is the ASGI app
- Userdata script auto-installs everything on EC2 boot (you configured this in the lecture)

---

#### PRACTICE TASK 5.1: Build the FastAPI Serving App

**What to Build:** Create a FastAPI app that loads the wine model and exposes `/predict` and `/health` endpoints.

**How:** Create `src/serve.py` with the code below. Run locally with uvicorn. Test with curl.

**Done When:** `curl localhost:8000/health` returns status:healthy. `/predict` returns wine class 0, 1, or 2.

```python
# src/serve.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title='Wine Classifier API', version='1.0')

# Load model
model = joblib.load('models/wine_model.pkl')
CLASS_NAMES = ['Class_0 Barolo', 'Class_1 Grignolino', 'Class_2 Barbera']

class WineInput(BaseModel):
    features: list

@app.get('/health')
def health():
    return {'status': 'healthy', 'model': 'wine-classifier-v1'}

@app.post('/predict')
def predict(wine: WineInput):
    # Validate input
    if len(wine.features) != 13:
        raise HTTPException(status_code=400, detail='Need exactly 13 features')
    
    # Predict
    arr = np.array(wine.features).reshape(1, -1)
    pred = int(model.predict(arr)[0])
    conf = float(model.predict_proba(arr)[0].max())
    
    return {
        'wine_class': pred, 
        'confidence': conf, 
        'class_name': CLASS_NAMES[pred]
    }
```

```bash
# Test locally
uvicorn src.serve:app --reload --port 8000

# Test health
curl http://localhost:8000/health

# Test prediction (13 wine feature values)
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]}'
```

---

#### PRACTICE TASK 5.2: Deploy to AWS EC2 with Userdata Script

**What to Build:** Launch an EC2 instance, use a Userdata script to auto-install the app on boot, test via public IP.

**How:** AWS Console → EC2 → Launch Instance → t2.micro, Ubuntu 22.04. Paste userdata in Advanced Details. Open port 8000 in Security Group.

**Done When:** `curl http://YOUR-EC2-PUBLIC-IP:8000/health` returns healthy from your laptop (not inside AWS).

```bash
#!/bin/bash
# EC2 Userdata script — paste into 'Advanced Details > User data' in AWS Console

apt-get update -y
apt-get install -y python3-pip python3-venv git

cd /home/ubuntu
git clone https://github.com/YOUR_USERNAME/mlops-project.git
cd mlops-project

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt gunicorn

# Download model from S3
aws s3 cp s3://YOUR-BUCKET/models/wine_model.pkl models/wine_model.pkl

# Start server with gunicorn (WSGI production server)
nohup gunicorn src.serve:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --log-file /var/log/mlapi.log &

echo 'API server started' >> /var/log/startup.log
```

---

## SECTION 6 — Docker + Kubernetes Deployment

### Section 6: Docker + Kubernetes Practice

The course walks through Dockerizing the Intent Classifier and deploying to Kubernetes. Mirror every step with your Wine Classifier model.

---

#### PRACTICE TASK 6.1: Dockerize Your Model API

**What to Build:** Write a Dockerfile, build the image, run it locally, push to Docker Hub or AWS ECR.

**How:** Create Dockerfile in project root. Build with `docker build`. Test with `docker run`. Push to registry.

**Done When:** `docker run -p 8000:8000 wine-classifier:v1` serves predictions. Image visible in your registry.

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

EXPOSE 8000

CMD ["gunicorn", "src.serve:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

```bash
# Build and test locally
docker build -t wine-classifier:v1 .
docker run -p 8000:8000 wine-classifier:v1
curl http://localhost:8000/health

# Push to Docker Hub
docker login
docker tag wine-classifier:v1 YOUR_USER/wine-classifier:v1
docker push YOUR_USER/wine-classifier:v1
```

---

#### PRACTICE TASK 6.2: Deploy to Kubernetes with Manifests

**What to Build:** Create Deployment + Service YAML manifests and deploy your containerized model to K8s.

**How:** Write `k8s/deployment.yaml` and `k8s/service.yaml`. Apply with `kubectl`. Verify pods are Running.

**Done When:** `kubectl get pods` shows wine-classifier pod Running. Port-forward works. Predictions return correctly.

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wine-classifier
  namespace: ml-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: wine-classifier
  template:
    metadata:
      labels:
        app: wine-classifier
    spec:
      containers:
      - name: wine-classifier
        image: YOUR_USER/wine-classifier:v1
        ports:
        - containerPort: 8000
        resources:
          requests: {cpu: 100m, memory: 256Mi}
          limits: {cpu: 500m, memory: 512Mi}
        readinessProbe:
          httpGet: {path: /health, port: 8000}
---
apiVersion: v1
kind: Service
metadata:
  name: wine-classifier-svc
  namespace: ml-serving
spec:
  selector: {app: wine-classifier}
  ports:
  - port: 80
    targetPort: 8000
```

```bash
kubectl create namespace ml-serving
kubectl apply -f k8s/deployment.yaml -n ml-serving
kubectl get pods -n ml-serving -w
kubectl port-forward svc/wine-classifier-svc 8080:80 -n ml-serving
curl http://localhost:8080/health
```

---

#### PRACTICE TASK 6.3: Add Ingress Controller

**What to Build:** Deploy NGINX Ingress Controller and route external traffic to your model via an Ingress resource.

**How:** Install ingress-nginx with Helm. Create `k8s/ingress.yaml`. Test with the ingress hostname.

**Done When:** `curl http://wine.local/health` returns 200. `kubectl get ingress` shows ADDRESS populated.

```bash
# Install NGINX Ingress Controller via Helm
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx -n ingress-nginx --create-namespace
```

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: wine-ingress
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - host: wine.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: wine-classifier-svc
            port: {number: 80}
```

```bash
kubectl apply -f k8s/ingress.yaml

# Add '127.0.0.1 wine.local' to /etc/hosts for local testing
```

---

## SECTION 7 — KServe

### Section 7: KServe Practice — Production Model Serving

KServe is the industry standard for serving ML models on Kubernetes. It handles auto-scaling, traffic splitting, canary deployments, and more. The course shows end-to-end demos for both a sample model and the Intent Classifier.

**Prerequisites Before Starting**
- Kubernetes cluster running (Kind or Minikube) — completed in Section 6
- Your model saved in S3 in sklearn joblib format
- Istio service mesh: required for KServe to work

---

#### PRACTICE TASK 7.1: Install KServe on Your Cluster

**What to Build:** Install KServe with all prerequisites: Istio, Cert-Manager, and KServe operator.

**How:** Follow the commands below in order. Each step must complete successfully before the next.

**Done When:** `kubectl get pods -n kserve-system` shows all pods Running. `kubectl get inferenceservices` returns empty list (ready).

```bash
# Step 1: Install Istio
curl -L https://istio.io/downloadIstio | sh -
cd istio-*/
export PATH=$PWD/bin:$PATH
istioctl install --set profile=demo -y

# Step 2: Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=90s

# Step 3: Install KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.12.0/kserve.yaml
kubectl wait --for=condition=ready pod -l control-plane=kserve-controller-manager \
  -n kserve-system --timeout=120s

# Verify
kubectl get pods -n kserve-system
kubectl get pods -n istio-system
```

---

#### PRACTICE TASK 7.2: Deploy Wine Model as KServe InferenceService

**What to Build:** Create an InferenceService YAML and deploy your Wine Classifier to KServe.

**How:** Save your model to S3 as `model.pkl`. Create `inferenceservice.yaml`. Deploy and test via Istio ingress.

**Done When:** `kubectl get inferenceservice wine-classifier` shows READY=True. Curl returns predictions via V2 protocol.

```bash
# Save model to S3 first (run this Python code locally)
# import boto3, joblib
# boto3.client('s3').upload_file('models/wine_model.pkl', 'YOUR-BUCKET', 'wine-model/model.pkl')
```

```yaml
# k8s/inferenceservice.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: wine-classifier
  namespace: kserve-test
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: s3://YOUR-BUCKET/wine-model
```

```bash
kubectl create namespace kserve-test
kubectl apply -f k8s/inferenceservice.yaml
kubectl get inferenceservice wine-classifier -n kserve-test -w

# Get Istio ingress IP
export INGRESS_IP=$(kubectl get svc istio-ingressgateway \
  -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test via KServe V2 protocol
curl -H 'Host: wine-classifier.kserve-test.example.com' \
  http://${INGRESS_IP}/v2/models/wine-classifier/infer \
  -d '{"inputs": [{"name": "input", "shape": [1,13], "datatype": "FP32",
       "data": [[14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065]]}]}'
```

---

## SECTION 8 — Amazon SageMaker

### Section 8: Amazon SageMaker Practice

SageMaker is AWS fully-managed ML platform. The course covers full production setup across multiple lectures. Practice each step in the order the course presents them.

| Practice Task | What to Do |
|---------------|------------|
| 8.1 — SageMaker Studio Setup | AWS Console → SageMaker → Studio → Create Domain → Quick Setup → Launch Studio |
| 8.2 — Submit Training Job | Use Python SDK SKLearn estimator to submit training job using S3 data |
| 8.3 — Register in Model Registry | Register trained model in SageMaker Model Registry with accuracy metrics |
| 8.4 — Deploy Real-Time Endpoint | Deploy registered model as real-time endpoint. Test with invoke_endpoint |
| 8.5 — Batch Transform | Run batch transform job to score a full CSV file of wine samples |
| 8.6 — IMPORTANT: Delete All Resources | Delete endpoint + model + endpoint config immediately to avoid AWS charges |

---

#### PRACTICE TASK 8.1: Train Wine Model as SageMaker Training Job

**What to Build:** Use SageMaker's sklearn container to train your model in the cloud with data from S3.

**How:** Upload `wine.csv` to S3. Write SageMaker-compatible `train.py` (reads from `/opt/ml/input/data`). Submit Training Job via Python SDK.

**Done When:** SageMaker console shows Training Job status COMPLETED. Model artifact appears in S3 output path.

```python
# sagemaker_train.py — run this locally to submit job to SageMaker

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

session = sagemaker.Session()
role = 'arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole'
bucket = session.default_bucket()

# Upload training data to S3
train_input = session.upload_data('data/raw/wine.csv', bucket=bucket, key_prefix='wine-data')

# Define estimator
estimator = SKLearn(
    entry_point='src/train_sm.py',
    role=role,
    framework_version='1.2-1',
    instance_type='ml.m5.large',
    instance_count=1,
    output_path='s3://' + bucket + '/wine-output/'
)

# Submit training job
estimator.fit({'train': train_input})
print('Training complete. Model URI:', estimator.model_data)
```

```python
# src/train_sm.py — SageMaker-compatible training script

import os
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-estimators', type=int, default=100)
    args, _ = parser.parse_known_args()
    
    # SageMaker passes data path via environment variable
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
    df = pd.read_csv(os.path.join(train_dir, 'wine.csv'))
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=args.n_estimators)
    model.fit(X_tr, y_tr)
    
    # SageMaker expects model saved to SM_MODEL_DIR
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
```

---

#### PRACTICE TASK 8.2: Deploy and Test a SageMaker Real-Time Endpoint

**What to Build:** Deploy your trained model as a live SageMaker endpoint and test predictions. Then delete it.

**How:** Use `estimator.deploy()` to create endpoint. Use `predictor.predict()` to test with real wine features. Then delete.

**Done When:** `predictor.predict()` returns class prediction. SageMaker console shows endpoint InService. Then shows Deleted.

```python
# Deploy endpoint (costs money per hour — delete when done!)
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='wine-classifier-endpoint'
)

# Test prediction
import numpy as np
sample = np.array([[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]])
result = predictor.predict(sample)
print('Prediction:', result)

# IMPORTANT: DELETE to stop charges
predictor.delete_endpoint()
print('Endpoint deleted — no more charges')
```

---

## SECTION 9 — Kubeflow Pipelines (59-min Tutorial)

### Section 9: Kubeflow Pipelines Practice

Kubeflow Pipelines lets you define your ML workflow as a graph of containerized steps. This is the gold standard for production ML pipelines. The course tutorial is 59 minutes — work through it carefully.

---

#### PRACTICE TASK 9.1: Install Kubeflow Pipelines on Your Cluster

**What to Build:** Install KFP standalone (lighter than full Kubeflow) on your Kind cluster. Open the dashboard UI.

**How:** Apply the KFP manifest. Wait for all pods. Port-forward the UI service. Open the dashboard.

**Done When:** KFP UI accessible at `localhost:8080`. Default pipelines visible. Can create new experiments from UI.

```bash
export KFP_VERSION=2.2.0

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$KFP_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$KFP_VERSION"

# Wait for all pods to be ready
kubectl wait pods -l application-crd-id=kubeflow-pipelines -n kubeflow \
  --for=condition=Ready --timeout=180s

# Open KFP UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &

# Open: http://localhost:8080
```

---

#### PRACTICE TASK 9.2: Build a 3-Component KFP Pipeline

**What to Build:** Write a pipeline with 3 Python components: `prepare_data`, `train_model`, `evaluate_model`. Submit to KFP UI.

**How:** Install kfp SDK. Define 3 `@component` functions. Connect them in a `@pipeline`. Compile to YAML. Submit.

**Done When:** KFP UI shows your pipeline run SUCCEEDED with all 3 components showing green checkmarks.

```bash
pip install kfp==2.7.0
```

```python
# pipeline.py — KFP v2 pipeline

from kfp import dsl
from kfp.dsl import component, Output, Input, Dataset, Model, Metrics
import kfp

@component(base_image='python:3.11-slim',
           packages_to_install=['scikit-learn', 'pandas'])
def prepare_data(dataset: Output[Dataset]):
    from sklearn.datasets import load_wine
    import pandas as pd
    
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.to_csv(dataset.path, index=False)

@component(base_image='python:3.11-slim',
           packages_to_install=['scikit-learn', 'pandas', 'joblib'])
def train_model(dataset: Input[Dataset], model: Output[Model], n_estimators: int = 100):
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(dataset.path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_tr, y_tr)
    
    joblib.dump(clf, model.path)

@component(base_image='python:3.11-slim',
           packages_to_install=['scikit-learn', 'pandas', 'joblib'])
def evaluate_model(dataset: Input[Dataset], model: Input[Model], metrics: Output[Metrics]):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(dataset.path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = joblib.load(model.path)
    acc = accuracy_score(y_te, clf.predict(X_te))
    
    metrics.log_metric('accuracy', acc)
    print('Accuracy: ' + str(round(acc, 4)))

@dsl.pipeline(name='Wine Classifier Pipeline')
def wine_pipeline(n_estimators: int = 100):
    data_op = prepare_data()
    train_op = train_model(dataset=data_op.outputs['dataset'], n_estimators=n_estimators)
    eval_op = evaluate_model(dataset=data_op.outputs['dataset'], model=train_op.outputs['model'])

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(wine_pipeline, 'wine_pipeline.yaml')
    
    client = kfp.Client(host='http://localhost:8080')
    run = client.create_run_from_pipeline_func(
        wine_pipeline, 
        arguments={'n_estimators': 150}
    )
    print('Pipeline submitted! Run ID: ' + run.run_id)
```

---

## SECTION 10 — CI/CD: GitHub Actions + DVC + Argo CD + KServe

### Section 10: CI/CD Practice — Full GitOps Pipeline

This is the capstone section (1h 22min). It ties everything together into a fully automated GitOps pipeline. This is what MLOps engineers build in real companies.

**The Full Flow You Are Building**
1. Developer pushes code to GitHub
2. GitHub Actions: runs tests, triggers DVC pipeline, checks accuracy threshold
3. If accuracy passes: builds Docker image, pushes to registry, updates K8s manifests
4. Argo CD detects manifest change, syncs to cluster, KServe deploys new model version

---

#### PRACTICE TASK 10.1: GitHub Actions CI — Train, Test, Build, Push

**What to Build:** Write a GitHub Actions workflow that trains model, checks accuracy, builds Docker image, and updates K8s manifests.

**How:** Create `.github/workflows/ci.yaml` with the code below. Add AWS secrets to GitHub repo settings. Push code to trigger.

**Done When:** GitHub Actions tab shows green checkmark. Push a bad model change — workflow fails on accuracy check.

```yaml
# .github/workflows/ci.yaml
name: ML CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  ml-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Pull DVC data from S3
        run: dvc pull
      
      - name: Run DVC pipeline (train model)
        run: dvc repro
      
      - name: Check accuracy threshold (must be >= 0.85)
        run: |
          python -c "
          import json
          with open('models/metrics.json') as f: m = json.load(f)
          acc = m['accuracy']
          print('Model accuracy: ' + str(acc))
          assert acc >= 0.85, 'FAIL: Accuracy below 0.85 threshold'
          print('PASS: Accuracy above threshold')"
      
      - name: Run unit tests
        run: pytest tests/ -v
      
      - name: Build and push Docker image
        if: github.ref == 'refs/heads/main'
        run: |
          docker build -t ${{ secrets.DOCKERHUB_USER }}/wine-classifier:${{ github.sha }} .
          docker login -u ${{ secrets.DOCKERHUB_USER }} -p ${{ secrets.DOCKERHUB_TOKEN }}
          docker push ${{ secrets.DOCKERHUB_USER }}/wine-classifier:${{ github.sha }}
      
      - name: Update K8s manifest with new image tag
        if: github.ref == 'refs/heads/main'
        run: |
          NEW_IMAGE="${{ secrets.DOCKERHUB_USER }}/wine-classifier:${{ github.sha }}"
          sed -i "s|image:.*wine-classifier.*|image: ${NEW_IMAGE}|" k8s/deployment.yaml
          git config user.email 'ci@github.com'
          git config user.name 'GitHub Actions'
          git add k8s/deployment.yaml
          git commit -m 'ci: update image tag to ${{ github.sha }}'
          git push
```

---

#### PRACTICE TASK 10.2: Argo CD — GitOps Continuous Delivery

**What to Build:** Install Argo CD on Kubernetes. Connect it to your GitHub repo. Watch it auto-deploy when manifests change.

**How:** Install Argo CD. Create an Application pointing to your `k8s/` folder. Change an image tag in Git → Argo CD auto-syncs.

**Done When:** Argo CD UI shows app as Synced + Healthy. Any change to `k8s/deployment.yaml` auto-deploys within 3 minutes.

```bash
# Install Argo CD
kubectl create namespace argocd
kubectl apply -n argocd -f \
  https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for Argo CD to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=argocd-server \
  -n argocd --timeout=120s

# Get initial admin password
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath='{.data.password}' | base64 -d

# Port-forward to Argo CD UI
kubectl port-forward svc/argocd-server -n argocd 8443:443 &

# Open: https://localhost:8443 (login: admin / password from above)

# In Argo CD UI: New App
# Application Name: wine-classifier
# Project: default
# Source Repo: https://github.com/YOUR_USER/mlops-project
# Source Path: k8s/
# Destination: https://kubernetes.default.svc
# Namespace: ml-serving
# Sync Policy: Automated (enable auto-sync, prune, self-heal)

# Test it: change image tag in k8s/deployment.yaml and push to GitHub
# Argo CD should detect the change and sync within 3 minutes
```

---

## MASTER CHECKLIST + INTERVIEW PREPARATION

### Master Checklist — Track Your Progress

Use this table to track completion. Write the date when you pass the 'Done When' condition for each task.

| Skill / Task | Date Completed |
|--------------|----------------|
| 1.1 — Project scaffold committed to GitHub with .gitignore + requirements.txt | |
| 1.2 — Wine model training script runs, saves .pkl, accuracy >= 0.90 | |
| 3.1 — DVC initialized, wine.csv tracked, .dvc file committed to Git | |
| 3.1 — DVC S3 remote configured, dvc push/pull works correctly | |
| 3.2 — DVC pipeline (dvc.yaml) has 2+ stages, dvc repro runs cleanly | |
| 4.1 — MLflow: 5 runs visible in UI, metrics compared, best run identified | |
| 4.1 — Best model registered in MLflow Model Registry with Staging stage | |
| 4.2 — MLflow server running on K8s with PostgreSQL backend | |
| 5.1 — FastAPI app: /health and /predict endpoints tested locally with curl | |
| 5.2 — EC2 deployment: API accessible via public IP from your laptop | |
| 6.1 — Docker image built, tested locally, pushed to Docker Hub or ECR | |
| 6.2 — K8s Deployment + Service: pod Running, port-forward works | |
| 6.3 — NGINX Ingress Controller: traffic routed to model via hostname | |
| 7.1 — KServe installed: all pods Running in kserve-system namespace | |
| 7.2 — KServe InferenceService: READY=True, predictions via V2 protocol work | |
| 8.1 — SageMaker Training Job: COMPLETED, model artifact in S3 output path | |
| 8.2 — SageMaker endpoint: InService, predictions tested, endpoint DELETED | |
| 9.1 — Kubeflow Pipelines: installed, UI accessible at localhost:8080 | |
| 9.2 — KFP 3-component pipeline: run SUCCEEDED, all components green | |
| 10.1 — GitHub Actions CI: triggers on push, fails on bad accuracy | |
| 10.2 — Argo CD: app Synced+Healthy, auto-deploys on manifest change | |
| **FINAL — Full GitOps**: push code → CI → Docker push → manifest update → Argo CD deploy | |

---

### Interview Preparation — How to Talk About Your Practice

The course has a dedicated interview preparation lecture. Here is how to frame each section of practice when answering real interview questions.

| Question | Answer Based on Your Practice Work |
|----------|------------------------------------|
| How do you version training data? | I use DVC with AWS S3 as remote storage. Each dataset version is a .dvc pointer file committed to Git, keeping code and data always in sync. Any experiment is reproducible by checking out the Git commit and running `dvc pull`. |
| How do you track ML experiments? | I use MLflow deployed on Kubernetes with PostgreSQL. Every run automatically logs parameters, metrics, and model artifact. I use the Model Registry to promote models through Staging to Production with full version history. |
| How do you serve models in production? | I have used three approaches: FastAPI with gunicorn on EC2 for simple APIs; K8s Deployment with Ingress for containerized workloads; and KServe InferenceService for production serving with auto-scaling and traffic splitting. |
| How do you automate model deployment? | Full GitOps with GitHub Actions and Argo CD. CI runs DVC pipeline, checks accuracy threshold, builds and pushes Docker image, then updates K8s manifests. Argo CD detects the change and auto-syncs the cluster. |
| What AWS ML services have you used? | Amazon SageMaker: submitted training jobs via Python SDK, registered models in Model Registry, deployed real-time inference endpoints, and ran batch transform jobs. Always cleaned up resources to control costs. |
| What is a Kubeflow Pipeline? | A workflow defined in Python with KFP SDK v2. Each step is a `@component` running in its own container. I built a 3-component pipeline (prepare, train, evaluate), compiled to YAML, and submitted to the KFP UI. |

---

### Daily Practice Schedule

| Week | Daily Focus (1-2 hours per day) | Goal |
|------|---------------------------------|------|
| Week 1 | Scaffold project, write train.py, push to GitHub, run model | Model working and committed to GitHub |
| Week 2 | DVC init + S3 remote + reproducible pipeline with dvc repro | `dvc repro` runs pipeline end to end |
| Week 3 | Local MLflow → K8s MLflow → PostgreSQL MLflow | MLflow UI on K8s showing 5+ tracked runs |
| Week 4 | FastAPI serve + gunicorn WSGI + EC2 Userdata deployment | API live on EC2 via public IP |
| Week 5 | Dockerfile + K8s manifests + Ingress Controller | Docker image in registry, pod Running in K8s |
| Week 6 | KServe install + InferenceService deploy + V2 protocol test | KServe serving predictions successfully |
| Week 7 | SageMaker: training job + endpoint + batch transform + delete | Full SageMaker workflow completed |
| Week 8 | Kubeflow Pipelines install + 3-component pipeline | Pipeline SUCCEEDED in KFP UI |
| Week 9 | GitHub Actions CI + Argo CD GitOps + end-to-end flow | Full GitOps pipeline working end to end |
| Week 10 | Integration pass + interview prep + write resume points | Can explain all projects in an interview |

---

**The Single Most Important Rule**

Every task in this guide has a 'Done When' check.

Do NOT move on until that check passes in your terminal or browser.

Your GitHub commit history is your portfolio. Commit after every working task.

**Start with Task 1.1 right now. Open your terminal and type: `mkdir mlops-project`**
```
