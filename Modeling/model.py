from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, weekofyear, year, lit, countDistinct, when, sum as sql_sum, monotonically_increasing_id, split
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Window
from pyspark.sql.functions import lag
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# Constants for application and data file paths
APP_NAME = "Chicago Crime Analysis"
CRIMES_CSV_PATH = "hdfs://wolf:9000/user/uhw4967/crime_data/Crimes_-_2001_to_present.csv"
IUCR_CSV_PATH = "hdfs://wolf:9000/user/uhw4967/crime_data/IUCR.csv"
POP_PATH = "hdfs://wolf:9000/user/uhw4967/crime_data/beatpop.txt"
RACE_PATH = "hdfs://wolf:9000/user/uhw4967/crime_data/beatrace.txt"
FS_PATH = "hdfs://wolf:9000/user/uhw4967/crime_data/beatfs.txt"

def load_pop_data(spark):
    """Load population data from txt file."""
    pop_df = spark.read.text(POP_PATH)
    pop_df = pop_df.withColumn("value", pop_df["value"].substr(2, 1000))  # Ignore first line
    pop_df = pop_df.withColumnRenamed("value", "temp_col")
    pop_df = pop_df.withColumn("Beat", split(col("temp_col"), " ").getItem(0)) \
                   .withColumn("Population", split(col("temp_col"), " ").getItem(1).cast("int")) \
                   .withColumn("Area", split(col("temp_col"), " ").getItem(2).cast("double")) \
                   .withColumn("population_density", col("Population") / col("Area")) \
                   .drop("temp_col")
    return pop_df

def load_race_data(spark):
    """Load race data from txt file."""
    race_df = spark.read.text(RACE_PATH)
    race_df = race_df.withColumn("value", race_df["value"].substr(2, 1000))  # Ignore first line
    race_df = race_df.withColumnRenamed("value", "temp_col")
    race_df = race_df.withColumn("Beat", split(col("temp_col"), " ").getItem(0)) \
                     .withColumn("White", split(col("temp_col"), " ").getItem(1).cast("double")) \
                     .withColumn("Hispanic", split(col("temp_col"), " ").getItem(2).cast("double")) \
                     .withColumn("Black", split(col("temp_col"), " ").getItem(3).cast("double")) \
                     .withColumn("Asian", split(col("temp_col"), " ").getItem(4).cast("double")) \
                     .withColumn("Mixed", split(col("temp_col"), " ").getItem(5).cast("double")) \
                     .withColumn("Other", split(col("temp_col"), " ").getItem(6).cast("double")) \
                     .drop("temp_col")
    return race_df


def load_fs_data(spark):
    """Load food stamp data from txt file."""
    fs_df = spark.read.text(FS_PATH)
    fs_df = fs_df.withColumn("value", fs_df["value"].substr(2, 1000))  # Ignore first line
    fs_df = fs_df.withColumnRenamed("value", "temp_col")
    fs_df = fs_df.withColumn("Beat", split(col("temp_col"), " ").getItem(0)) \
                 .withColumn("FoodStamps", split(col("temp_col"), " ").getItem(1).cast("int")) \
                 .drop("temp_col")
    return fs_df
