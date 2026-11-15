#!/bin/bash
# One-time setup (safe to re-run; it resets the Airflow database):
# writes .env with your user id, creates the folders Airflow mounts, builds the custom image
# and initialises the database with the admin user (airflow / airflow).
set -e
cd "$(dirname "$0")"

# Stop anything from a previous run and remove its database volume
docker compose down -v --remove-orphans

# Folders mounted into the containers (config/airflow.cfg is kept, unlike the original script)
mkdir -p ./logs ./plugins ./config ./output

# Files written by the containers will be owned by you
echo "AIRFLOW_UID=$(id -u)" > .env

docker compose build
docker compose up airflow-init
