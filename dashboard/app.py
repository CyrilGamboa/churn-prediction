import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

# Toujours obtenir le chemin absolu du projet
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "app", "model")

st.write("Base directory utilisé :", BASE_DIR)
st.write("Model directory utilisé :", MODEL_DIR)

# API_URL = "http://localhost:8001"
API_URL = "http://churn-api:8000"  # ⚠️ à adapter selon ton docker-compose

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

st.title("Churn Prediction Dashboard")

# --- Onglets
tabs = st.tabs(["Prédiction Client", "Métriques Globales", "Analyse Modèle"])

# ===============================
# 1. Onglet Prédiction Client
# ===============================
with tabs[0]:
    st.header("Prédire le churn d’un client")

    # Formulaire
    with st.form("prediction_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (mois)", min_value=0, max_value=72, value=12)
        PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
        MultipleLines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox(
            "OnlineSecurity", ["Yes", "No", "No internet service"]
        )
        OnlineBackup = st.selectbox(
            "OnlineBackup", ["Yes", "No", "No internet service"]
        )
        DeviceProtection = st.selectbox(
            "DeviceProtection", ["Yes", "No", "No internet service"]
        )
        TechSupport = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox(
            "StreamingMovies", ["Yes", "No", "No internet service"]
        )
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "PaymentMethod",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        MonthlyCharges = st.number_input(
            "MonthlyCharges", min_value=0.0, max_value=200.0, value=70.0
        )
        TotalCharges = st.number_input(
            "TotalCharges", min_value=0.0, max_value=10000.0, value=1000.0
        )
        threshold = st.slider("Seuil de classification", 0.0, 1.0, 0.4, 0.05)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        payload = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        }
        r = requests.post(f"{API_URL}/predict?threshold={threshold}", json=payload)
        if r.status_code == 200:
            res = r.json()
            st.success(f"Churn : **{res['churn']}** (proba={res['probability']})")
        else:
            st.error("Erreur API")

# ===============================
# 2. Onglet Métriques globales
# ===============================
with tabs[1]:
    st.header("Métriques globales")

    threshold = st.slider("Seuil", 0.0, 1.0, 0.4, 0.05, key="metrics")
    r = requests.get(f"{API_URL}/metrics?threshold={threshold}")
    if r.status_code == 200:
        metrics = r.json()

        # --- Explications pour un public novice ---
        st.subheader("Explications des métriques 📊")

        st.markdown(
            f"""
        - **threshold (seuil)** : {metrics['threshold']}  
          C’est la limite à partir de laquelle on considère un client comme « en churn ».  
          Par exemple, si la probabilité prédite = 0.45 et le seuil = 0.40 → le modèle dit **churn**.

        - **precision (précision)** : {metrics['precision']}  
          Parmi tous les clients prédits comme churn, quelle proportion l’était vraiment ?  
          (Évite les faux positifs → ne pas alerter inutilement sur des clients fidèles).

        - **recall (rappel)** : {metrics['recall']}  
          Parmi tous les vrais clients en churn, combien le modèle en a détecté ?  
          (Évite les faux négatifs → ne pas manquer un client qui part).

        - **f1_score** : {metrics['f1_score']}  
          Moyenne entre précision et rappel (équilibre entre les deux).  
          Utile quand on veut **équilibrer** la détection et la fiabilité.

        - **accuracy (exactitude)** : {metrics['accuracy']}  
          Proportion globale de prédictions correctes (tous clients confondus).

        - **pr_auc (aire sous la courbe précision-rappel)** : {metrics['pr_auc']}  
        Mesure la qualité globale du modèle à identifier les clients en churn,  
          indépendamment d’un seuil fixe.
        """
        )

        st.subheader("Valeurs obtenues")
        st.json(metrics)

    else:
        st.warning("Impossible de récupérer les métriques")


# ===============================
# 3. Onglet Analyse Modèle
# ===============================
with tabs[2]:
    st.header("Analyse du modèle")

    # --- Feature importances ---
    try:
        df_importance = pd.read_csv(os.path.join(MODEL_DIR, "feature_importances.csv"))
        fig = px.bar(
            df_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importances (XGBoost)",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(
            f"Importances des features non disponibles : {e} (génère-les après entraînement)"
        )

    # --- Distribution des probabilités ---
    try:
        df_test = pd.read_csv(os.path.join(MODEL_DIR, "test_set.csv"))
        st.write("Distribution des probabilités de churn (jeu de test)")

        # ⚠️ cet endpoint doit être implémenté côté API
        r = requests.get(f"{API_URL}/predict_proba_all")
        if r.status_code == 200:
            y_proba = r.json()
            df_test["proba"] = y_proba
            fig = px.histogram(
                df_test,
                x="proba",
                nbins=30,
                title="Distribution des probabilités de churn",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Impossible de récupérer les probabilités depuis l’API")
    except Exception as e:
        st.info(f"Probabilités non disponibles : {e}")
