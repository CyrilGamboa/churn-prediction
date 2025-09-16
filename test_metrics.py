# test_metrics.py
import requests

API_URL = "http://localhost:8001"

# Payload d'exemple (un client fictif)
payload = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 89.10, "TotalCharges": 1068.20
}

# 1. Test de santé
print("=== /health ===")
r = requests.get(f"{API_URL}/health")
print(r.status_code, r.json())

# 2. Test de prédiction
print("\n=== /predict ===")
r = requests.post(f"{API_URL}/predict?threshold=0.40", json=payload)
print(r.status_code, r.json())

# 3. Test des métriques globales
print("\n=== /metrics ===")
r = requests.get(f"{API_URL}/metrics?threshold=0.40")
print(r.status_code, r.json())
