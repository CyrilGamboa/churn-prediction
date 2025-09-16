# test_latency.py
import httpx, time, statistics as stats

payload = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.10,
    "TotalCharges": 1068.20,
}
URL = "http://localhost:8000/predict?threshold=0.4"

lat = []
with httpx.Client(timeout=5.0) as client:
    # Warm-up
    for _ in range(3):
        client.post(URL, json=payload)

    for i in range(20):
        t0 = time.perf_counter()
        r = client.post(URL, json=payload)
        dt_ms = (time.perf_counter() - t0) * 1000
        lat.append(dt_ms)
        print(f"Requête {i+1}: {dt_ms:.2f} ms · code={r.status_code}")

lat_sorted = sorted(lat)
print(
    f"\nP50: {lat_sorted[len(lat)//2]:.2f} ms | Moyenne: {stats.mean(lat):.2f} ms | Min: {min(lat):.2f} ms | Max: {max(lat):.2f} ms"
)
