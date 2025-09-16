# ==============================
# 📌 Makefile - Churn Prediction Project
# ==============================

PROJECT_NAME=churn_api

# --- Commandes principales ---

# Lancer les conteneurs
up:
	docker-compose up

# Reconstruire les images et lancer les conteneurs
build:
	docker-compose up --build

# Arrêter les conteneurs
down:
	docker-compose down

# Arrêter et supprimer volumes + orphelins (reset complet)
clean:
	docker-compose down -v --remove-orphans

# Vérifier les logs de l'API
logs-api:
	docker logs -f churn-api

# Vérifier les logs du dashboard
logs-dashboard:
	docker logs -f churn-dashboard

# --- Commandes utilitaires ---

# Vérifier l'état des conteneurs
ps:
	docker ps

# Accéder au shell dans le conteneur API
shell-api:
	docker exec -it churn-api /bin/bash

# Accéder au shell dans le conteneur Dashboard
shell-dashboard:
	docker exec -it churn-dashboard /bin/bash

# Nettoyer les images inutilisées
prune:
	docker system prune -f
