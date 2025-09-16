import time
import logging
from fastapi import FastAPI, Query, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, average_precision_score

# Config logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("churn_api")

# Chargement du modèle
MODEL_PATH = "app/model/xgb_churn_pipeline.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}. Exécutez train.py au préalable.")
model = joblib.load(MODEL_PATH)
logger.info("✅ Modèle chargé avec succès")

# Chargement du jeu de test (si disponible)
try:
    TEST_PATH = "app/model/test_set.csv"
    df_test = pd.read_csv(TEST_PATH)
    X_test = df_test.drop(columns=["Churn"])
    y_test = df_test["Churn"]
    logger.info(f"✅ Jeu de test chargé depuis {TEST_PATH} ({X_test.shape})")
except Exception:
    df_test, X_test, y_test = None, None, None
    logger.warning("⚠️ Aucun jeu de test trouvé pour /metrics")

# Schéma d'entrée
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Création de l'app
app = FastAPI(default_response_class=ORJSONResponse)

# Middleware pour loguer la latence
@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} - {duration:.2f} ms - status {response.status_code}")
    return response

# Warm-up
@app.on_event("startup")
def warmup():
    sample = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 50.0, "TotalCharges": 50.0
    }])
    try:
        _ = model.predict_proba(sample)
        logger.info("Warm-up terminé")
    except Exception as e:
        logger.error(f"Warm-up échoué : {e}")

# Endpoint santé
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint prédiction
@app.post("/predict")
def predict(features: CustomerFeatures, threshold: float = Query(0.40, ge=0.0, le=1.0)):
    input_df = pd.DataFrame([features.dict()])
    prob = float(model.predict_proba(input_df)[0][1])
    return {
        "churn": bool(prob >= threshold),
        "probability": round(prob, 3),
        "threshold": threshold
    }

# Endpoint métriques globales
@app.get("/metrics")
def metrics(threshold: float = Query(0.40, ge=0.0, le=1.0)):
    if X_test is None or y_test is None:
        return {"error": "Aucun jeu de test n'est disponible pour calculer les métriques."}

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    pr_auc = average_precision_score(y_test, y_proba)

    return {
        "threshold": threshold,
        "precision": round(float(report["1"]["precision"]), 3),
        "recall": round(float(report["1"]["recall"]), 3),
        "f1_score": round(float(report["1"]["f1-score"]), 3),
        "accuracy": round(float(report["accuracy"]), 3),
        "pr_auc": round(float(pr_auc), 3)
    }

# Endpoint pour renvoyer toutes les probabilités du jeu de test
@app.get("/predict_proba_all")
def predict_proba_all():
    if X_test is None or model is None:
        return {"error": "Le jeu de test ou le modèle n'est pas disponible."}

    try:
        # Probabilité de churn (classe positive = 1)
        y_proba = model.predict_proba(X_test)[:, 1]
        return y_proba.tolist()  # conversion en liste pour JSON
    except Exception as e:
        return {"error": str(e)}

