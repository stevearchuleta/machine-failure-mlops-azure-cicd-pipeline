```md
# MACHINE FAILURE PREDICTION - AZURE MLOPS PIPELINE

## 1. Project Overview
This project provides an **MLOps pipeline for predicting machine failures** in industrial equipment using a machine learning model. The model is trained using an **XGBoost Classifier**, which analyzes operational data such as **temperature, pressure, vibration levels, and machine load** to determine the likelihood of failure.

The pipeline is fully automated using **Azure Machine Learning (Azure ML)** for model training, evaluation, and deployment. **GitHub Actions is used for CI/CD automation**, ensuring that each model update is seamlessly integrated into the pipeline.

### Dataset Description
- **File Name:** `machine_failure_data.csv`
- **Columns:**  
  - `Failure`: Target variable (binary, 1 = Machine Failure, 0 = No Failure).  
  - `Temperature`: Operating temperature of the machine.  
  - `Pressure`: Internal pressure level.  
  - `Vibration`: Vibration intensity level of the machine.  
  - `Load`: Machine workload as a percentage of capacity.  
  - `RPM`: Revolutions per minute of the machine's motor.  
- **Data Size:** Approximately **100,000 records** collected from industrial monitoring systems.

---

## 2. Installation and Setup

### 1️⃣ Install Dependencies
Run the `setup.sh` script to install required Python dependencies:
```sh
bash setup.sh
```

### 2️⃣ Ensure Azure ML CLI is Installed
```sh
az extension add -n ml -y
```

---

## 3. Pipeline Workflow
The MLOps pipeline consists of the following key stages:

### 1️⃣ Preprocessing (`prep.py`)
- Loads `machine_failure_data.csv` and cleans the data.
- Handles **missing values** using median imputation.
- Encodes categorical features (if applicable).
- Scales numerical features (`Temperature`, `Pressure`, `Vibration`, `Load`, `RPM`).

### 2️⃣ Training (`train.py`)
- Trains an **XGBoost Classifier** model on the preprocessed dataset.
- Uses **Accuracy, Precision, Recall, F1-score, and AUC-ROC** as evaluation metrics.

### 3️⃣ Evaluation (`train.py`)
- The model’s performance is **logged in MLflow**.
- Evaluation metrics can be accessed in **Azure ML Studio**.

### 4️⃣ Model Registration (`register.py`)
- Saves the trained model to **Azure ML** for deployment.
- Stores metadata about the registered model.

**Azure ML orchestrates job execution**, ensuring each step runs in the correct sequence.

---

## 4. How to Run the Pipeline

### Submit the MLOps Pipeline to Azure ML
Use the following command to submit the pipeline job:
```sh
bash run-job.sh mlops/azureml/train/newpipeline.yml machine-failure-experiment
```
This command **submits an Azure ML job**, triggering preprocessing, training, evaluation, and model registration.

---

## 5. Folder and File Structure
This project follows **Azure MLOps best practices** with the following directory structure:

- `.github/workflows/` → Contains **CI/CD pipeline definitions** (`ci.yml`, `deploy-model-training-pipeline-classical.yml`).
- `data/` → Stores raw dataset (`machine_failure_data.csv`).
- `data-science/src/` → Contains machine learning scripts (`prep.py`, `train.py`, `register.py`).
- `mlops/azureml/train/` → Stores Azure ML pipeline definitions (`newpipeline.yml`, `train.yml`).
- `jobs/pipeline/` → Contains `pipeline.yml`, which outlines the entire MLOps workflow.
- `requirements.txt` → Lists Python dependencies.
- `setup.sh` → Installs **Azure ML CLI tools**.
- `run-job.sh` → Submits the MLOps pipeline job.

---

## 6. CI/CD Automation with GitHub Actions
This project uses **GitHub Actions** to automate **model training and deployment**.

- `ci.yml` → Runs **pre-commit checks, unit tests, and formatting validation**.
- `deploy-model-training-pipeline-classical.yml` → Deploys the model to **Azure ML Pipelines**.
- Every push to `main` **triggers a training pipeline**.

---

## 7. Additional Notes
- This project follows **MLOps best practices for Azure**.
- Future improvements may include **hyperparameter tuning and real-time model monitoring**.
- To monitor jobs, navigate to **Azure ML Studio** and check job logs.
```