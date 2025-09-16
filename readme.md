# 📊 Customer Churn Prediction – Télécoms
*(FR/EN bilingual README)*

## 🇫🇷 Présentation du projet
Ce projet vise à prédire le **churn client** (désabonnement) dans le secteur des télécommunications.  
L’objectif est de développer :  
- un **modèle de machine learning** optimisé pour le rappel (minimiser les clients quittant non détectés),  
- une **API FastAPI** permettant d’exposer le modèle en production,  
- une **interface Streamlit** pour tester facilement des prédictions,  
- un **déploiement via Docker Compose**.  

### Données
- Dataset : **Telco Customer Churn (Kaggle)**  
- Variables : démographiques (genre, âge), services souscrits, facturation, durée d’abonnement, etc.  
- Cible : `Churn` (Oui/Non).  

### Modèle
- **XGBoost Classifier**, optimisé par **GridSearchCV**.  
- Meilleurs hyperparamètres trouvés :  
  - `learning_rate=0.01`  
  - `max_depth=3`  
  - `n_estimators=100`  
  - etc.  

### Résultats (seuil 0.5)
- Accuracy : **72 %**  
- Recall (classe churn) : **82 %**  
- Precision (classe churn) : **48 %**  
- F1-score (classe churn) : **61 %**  

⚡ L’accent est mis sur le **recall** afin de capter un maximum de clients en risque de churn.  

---

## 🇬🇧 Project Overview
This project focuses on predicting **customer churn** in the telecom industry.  
The goal is to build:  
- a **machine learning model** optimized for recall (to minimize undetected churners),  
- a **FastAPI service** to expose the model,  
- a **Streamlit client app** to test predictions,  
- a **Docker Compose deployment** for reproducibility.  

### Dataset
- Source: **Telco Customer Churn (Kaggle)**  
- Features: demographics, subscribed services, billing, tenure, etc.  
- Target: `Churn` (Yes/No).  

### Model
- **XGBoost Classifier**, tuned via **GridSearchCV**.  
- Best parameters include:  
  - `learning_rate=0.01`  
  - `max_depth=3`  
  - `n_estimators=100`  
  - etc.  

### Results (threshold 0.5)
- Accuracy: **72 %**  
- Recall (churn class): **82 %**  
- Precision (churn class): **48 %**  
- F1-score (churn class): **61 %**  

⚡ Priority is set on **recall** to detect as many at-risk customers as possible.  

---

## ⚙️ Architecture
- **`train.py`** : Entraîne le modèle, sauvegarde le pipeline + jeu de test.  
- **`main.py`** : API FastAPI (`/predict`, `/metrics`, `/health`).  
- **`app.py`** : Interface utilisateur Streamlit.  
- **`requirements.txt`** : Dépendances Python.  
- **`Dockerfile` et `docker-compose.yml`** : Conteneurisation.  

---

## 🚀 Installation & Lancement

### 1. Cloner le projet
```bash
git clone https://github.com/<username>/churn_api.git
cd churn_api
```

### 2. Entraîner le modèle
```bash
python train.py
```
➡️ génère :  
- `app/model/xgb_churn_pipeline.pkl`  
- `app/model/test_set.csv`  

### 3. Lancer avec Docker Compose
```bash
docker-compose up --build
```

### 4. Tester l’API
Santé :
```bash
curl http://localhost:8001/health
```

Prédiction :
```bash
curl -X POST "http://localhost:8001/predict?threshold=0.4" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
       "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
       "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
       "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
       "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check", "MonthlyCharges": 89.10, "TotalCharges": 1068.20
     }'
```

### 5. Accéder au dashboard Streamlit
👉 [http://localhost:8501](http://localhost:8501)  

---

## 📌 Améliorations futures
- 🔧 Ajouter des visualisations dans Streamlit (distribution des probabilités, importance des features).  
- 📈 Monitoring du modèle (drift, métriques en continu).  
- ☁️ Déploiement sur cloud (AWS/GCP/Azure).  

