# Custom Airflow image with the lab's packages baked in, instead of installing them with
# _PIP_ADDITIONAL_REQUIREMENTS every time a container starts (Airflow's recommended way).
FROM apache/airflow:3.3.2-python3.12

COPY requirements.txt /requirements.txt
# Installing apache-airflow at the same version stops pip from upgrading or downgrading it
RUN pip install --no-cache-dir "apache-airflow==3.3.2" -r /requirements.txt
