import os
from datetime import datetime, timedelta
from urllib.parse import urlencode

from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import requests
import zipfile
from io import BytesIO



def create_tables():
    logger = logging.getLogger(__name__)

    try:
        import clickhouse_connect

        logger.info("Connecting to ClickHouse...")

        client = clickhouse_connect \
            .get_client(host="clickhouse",
                        port=8123,
                        username="default",
                        password="clickhouse")

        client.command("""
                    CREATE TABLE IF NOT EXISTS default.top_regions (
                        region String,
                        count_addresses Int32
                    ) ENGINE = MergeTree()
                    ORDER BY count_addresses
                """)
        client.command("TRUNCATE TABLE IF EXISTS default.top_regions")
        logger.info("Table ready 'Top 10 areas with the largest number of objects'")


        client.command("""
                    CREATE TABLE IF NOT EXISTS default.top_cities (
                        locality_name String,
                        count_addresses Int32
                    ) ENGINE = MergeTree()
                    ORDER BY count_addresses
                """)
        client.command("TRUNCATE TABLE IF EXISTS default.top_cities")
        logger.info("Table ready 'Top 10 cities with the largest number of sites'")

        client.command("""
                    CREATE TABLE IF NOT EXISTS default.max_min_area (
                        region String,
                        max_square Float32,
                        min_square Float32
                    ) ENGINE = MergeTree()
                    ORDER BY max_square
                """)
        client.command("TRUNCATE TABLE IF EXISTS default.max_min_area")
        logger.info("Table ready 'Buildings with a maximum and minimum area within each area'")


        client.command("""
                    CREATE TABLE IF NOT EXISTS default.decade_buildings (
                        decade Int32,
                        count Int32
                    ) ENGINE = MergeTree()
                    ORDER BY decade
                """)
        client.command("TRUNCATE TABLE IF EXISTS default.decade_buildings")
        logger.info("Table ready 'Number of buildings by decade'")


    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise


def get_data():
    logger = logging.getLogger(__name__)

    try:
        base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
        public_key = "https://disk.yandex.ru/d/bhf2M8C557AFVw"

        response = requests.get(base_url + urlencode(dict(public_key=public_key)))
        response.raise_for_status()
        download_url = response.json()["href"]

        archive_response = requests.get(download_url)
        archive_response.raise_for_status()

        def extract_zip(content, extract_to="data"):
            with zipfile.ZipFile(BytesIO(content), "r") as zip_ref:
                zip_ref.extractall(extract_to)

                for file_name in zip_ref.namelist():
                    file_path = os.path.join(extract_to, file_name)
                    if file_name.endswith(".zip") and os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            extract_zip(f.read(), extract_to)
                        os.remove(file_path)

        extract_zip(archive_response.content)
        logger.info("All archives have been successfully unpacked to the data folder")

    except Exception as e:
        logger.error(f"Error: {e}")


def transform_load():
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as sf
    import clickhouse_connect

    logger = logging.getLogger(__name__)

    client = clickhouse_connect \
        .get_client(host="clickhouse",
                    port=8123,
                    username="default",
                    password="clickhouse")

    spark = SparkSession.builder.getOrCreate()
    logger.info("SparkSession created")

    raw_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("encoding", "UTF-16")
        .csv("data/russian_houses.csv")
    )

    total_rows = raw_df.count()
    logger.info(f"Количество строк: {total_rows}")

    df = raw_df.withColumn("maintenance_year_clean", sf.regexp_extract("maintenance_year", "(\\d+)", 1)) \
        .withColumn("population_clean", sf.regexp_extract("population", "(\\d+)", 1)) \
        .withColumn("maintenance_year", sf.when(sf.col("maintenance_year_clean") != "",
                                                sf.col("maintenance_year_clean").cast("int")) \
                    .otherwise(None)) \
        .withColumn("population", sf.when(sf.col("population_clean") != "",
                                          sf.col("population_clean").cast("int")) \
                    .otherwise(None)) \
        .withColumn("square", sf.when(sf.col("square").isin("—", "-", ""), None) \
                    .otherwise(sf.regexp_replace(sf.col("square"), "[^0-9.]", "").cast("double"))) \
        .withColumn("decade", ((sf.col("maintenance_year") / 10).cast("int") * 10)) \
        .drop("maintenance_year_clean", "population_clean")

    df = df.dropna()

    logger.info("The data has been cleared")

    df = df.filter((df.decade >= 1950) & (df.decade <= 2026))

    average_year_of_buildings = df.agg(sf.avg(df.maintenance_year).alias("average_year_of_buildings"))
    logger.info(f"The average year of construction of buildings: {int(average_year_of_buildings.collect()[0][0])}")

    median_year_of_buildings = df.agg(sf.median(df.maintenance_year).alias("median_year_of_buildings"))
    logger.info(f"The median year of construction of buildings: {int(median_year_of_buildings.collect()[0][0])}")


    top_region = df.select(["region", "address"]) \
        .groupby("region") \
        .agg(sf.count("address").alias("count_addresses")) \
        .orderBy("count_addresses", ascending=False) \
        .limit(10)

    pandas_df = top_region.toPandas()
    client.insert_df("top_regions", pandas_df)
    logger.info("The top 10 regions with the largest number of objects have been uploaded.")


    top_cities = df.select(["locality_name", "address"]) \
        .groupby("locality_name") \
        .agg(sf.count("address").alias("count_addresses")) \
        .orderBy("count_addresses", ascending=False) \
        .limit(10)

    pandas_df = top_cities.toPandas()
    client.insert_df("top_cities", pandas_df)
    logger.info("The top 10 cities with the largest number of objects have been uploaded.")


    max_min_square = df.select(["region", "square", "square"]) \
        .groupby("region") \
        .agg(sf.max("square").alias("max_square"), sf.min("square").alias("min_square"))

    pandas_df = max_min_square.toPandas()
    client.insert_df("max_min_area", pandas_df)
    logger.info("The buildings with the maximum and minimum area within each area are loaded.")


    cnt_number_by_decade = df.select(["decade", "house_id"]) \
        .groupby("decade") \
        .agg(sf.count("house_id").alias("count")) \
        .orderBy("decade")

    pandas_df = cnt_number_by_decade.toPandas()
    client.insert_df("decade_buildings", pandas_df)
    logger.info("The number of buildings loaded by decade.")

    result = client.query("SELECT * FROM max_min_area WHERE min_square > 60 ORDER BY max_square DESC LIMIT 25")
    headers = result.column_names
    print("-" * 67)
    print(f"{headers[0]:<40} {headers[1]:<15} {headers[2]:<15}")
    print("-" * 67)

    for row in result.result_rows:
        print(f"{row[0]:<40} {row[1]:<15.2f} {row[2]:<15.2f}")
    print("-" * 67)

    spark.stop()


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="spark_to_clickhouse_production",
    default_args=default_args,
    schedule_interval=None,  # Или поставьте расписание: "0 */6 * * *" для каждых 6 часов
    catchup=False,
    description="Production Spark to ClickHouse pipeline",
    tags=["spark", "clickhouse"],
) as dag:

    download_data_job = PythonOperator(
        task_id="download_data",
        python_callable=get_data,
    )

    create_tables_job = PythonOperator(
        task_id="create_tables",
        python_callable=create_tables,
    )

    transform_load_job = PythonOperator(
        task_id="transform_load",
        python_callable=transform_load,
    )


create_tables_job >> download_data_job >> transform_load_job
