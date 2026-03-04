FROM apache/airflow:2.9.1-python3.11

USER root

RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean

COPY jars /opt/airflow/jars

USER airflow

RUN pip install pyspark==3.4.3 clickhouse-connect pandas requests

