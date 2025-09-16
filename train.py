# train.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import kagglehub

# Download latest version
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

# === 1. Chargement des données ===
#df = pd.read_csv("telco_churn.csv")  # !! a adapter selon le chemin du fichier
df = pd.read_csv(f"{path}/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Nettoyage
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)
df.drop(columns=["customerID"], inplace=True)

# Cible binaire
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Features
num_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
cat_features = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
]

X = df[num_features + cat_features]
y = df["Churn"]

# === 2. Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Poids de classe auto
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# === 3. Pipeline ===
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False
    ))
])

# === 4. Optimisation (GridSearch) ===
param_grid = {
    "classifier__max_depth": [3, 4],
    "classifier__learning_rate": [0.01, 0.1],
    "classifier__n_estimators": [100, 200],
    "classifier__subsample": [0.7, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0],
    "classifier__min_child_weight": [1, 5],
    "classifier__gamma": [0, 1]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best recall (CV):", grid.best_score_)

# === 5. Évaluation finale ===
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("\n=== Rapport classification (seuil 0.5) ===")
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))



# === Sauvegarde des importances des features ===
importances_path = "app/model/feature_importances.csv"

# Récupération des noms des features après préprocessing
feature_names = (
    num_features +
    list(best_model.named_steps["preprocessor"]
         .named_transformers_["cat"]
         .get_feature_names_out(cat_features))
)

# Récupération des importances depuis XGBoost
xgb_model = best_model.named_steps["classifier"]
importances = xgb_model.feature_importances_

# Sauvegarde dans un CSV
importances_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

importances_df.to_csv(importances_path, index=False)
print(f"Importances sauvegardées dans {importances_path}")






# === 6. Sauvegardes ===
os.makedirs("app/model", exist_ok=True)

# Sauvegarde du modèle
joblib.dump(best_model, "app/model/xgb_churn_pipeline.pkl")

# Sauvegarde du jeu de test
df_test = pd.concat([X_test, y_test], axis=1)
df_test.to_csv("app/model/test_set.csv", index=False)

print("\nModèle et test_set sauvegardés dans app/model/")
