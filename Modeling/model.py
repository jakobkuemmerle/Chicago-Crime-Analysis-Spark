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

def merge_extra_data(pop_df, race_df, fs_df):
    """Merge population, race, and food stamp data on the column 'Beat'."""
    merged_df = pop_df.join(race_df, on="Beat").join(fs_df, on="Beat")
    merged_df = merged_df.withColumn("food_stamp_perc", col("FoodStamps") / col("Population") * 100)
    merged_df = merged_df.drop("Population").drop("Area")
    return merged_df

DATE_FORMAT = 'MM/dd/yyyy hh:mm:ss a'

# List of violent and non-violent crimes
VIOLENT_CRIMES = [
    "OFFENSE INVOLVING CHILDREN", "PUBLIC PEACE VIOLATION", "ARSON", "ASSAULT", "BATTERY", "ROBBERY",
    "HUMAN TRAFFICKING", "SEX OFFENSE", "CRIMINAL DAMAGE", "KIDNAPPING", "INTERFERENCE WITH PUBLIC OFFICER"
]

NON_VIOLENT_CRIMES = [
    "OBSCENITY", "OTHER OFFENSE", "GAMBLING", "CRIMINAL TRESPASS", "LIQUOR LAW VIOLATION",
    "PUBLIC INDECENCY", "INTIMIDATION", "PROSTITUTION", "DECEPTIVE PRACTICE",
    "CONCEALED CARRY LICENSE VIOLATION", "NARCOTICS", "NON-CRIMINAL", "WEAPONS VIOLATION",
    "OTHER NARCOTIC VIOLATION"
]

def initialize_spark_session():
    """Initialize Spark session."""
    spark = SparkSession.builder.appName(APP_NAME).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_crime_data(spark):
    """Load crime data from CSV file."""
    crime_df = spark.read.csv(CRIMES_CSV_PATH, header=True)
    crime_df = crime_df.withColumn("Date", to_date(col("Date"), DATE_FORMAT))
    crime_df = crime_df.withColumn("Beat", col("Beat").cast("integer"))

    return crime_df

def load_iucr_data(spark):
    """Load IUCR data from CSV file."""
    iucr_df = spark.read.csv(IUCR_CSV_PATH, header=True)
    iucr_df = iucr_df.withColumnRenamed("IUCR", "IUCR_other")
    return iucr_df

def feature_engineering(crime_df, iucr_df):
    """Perform feature engineering on the data."""
    joined_df = crime_df.join(iucr_df, crime_df.IUCR == iucr_df.IUCR_other, 'inner')
    joined_df = joined_df.withColumn("week_of_year", weekofyear(col("Date")))
    joined_df = joined_df.withColumn("year", year(col("Date")))

    indexer = StringIndexer(inputCol="Beat", outputCol="BeatIndex")
    joined_df = indexer.fit(joined_df).transform(joined_df)

    joined_df = joined_df.withColumn("violent_crime", when(col("PRIMARY DESCRIPTION").isin(VIOLENT_CRIMES), 1).otherwise(0))
    joined_df = joined_df.withColumn("non_violent_crime", when(col("PRIMARY DESCRIPTION").isin(NON_VIOLENT_CRIMES), 1).otherwise(0))
    joined_df = joined_df.withColumn("Arrest", when(col("Arrest") == True, 1).otherwise(0))
    joined_df = joined_df.withColumn("Domestic", when(col("Domestic") == True, 1).otherwise(0))
    return joined_df

def aggregate_data(joined_df):
    """Aggregate crime data by beat, year, and week of the year."""
    aggregated_df = joined_df.groupBy("Beat", "BeatIndex", "year", "week_of_year") \
        .agg(
            sql_sum(when(col("violent_crime") == lit(1), 1)).alias("total_violent_crimes"),
            sql_sum(when(col("non_violent_crime") == lit(1), 1)).alias("total_non_violent_crimes"),
            sql_sum(col("Arrest")).alias("total_arrests"),
            sql_sum(col("Domestic")).alias("total_domestic_crimes"),
            countDistinct("District").alias("num_districts"),
            countDistinct("Ward").alias("num_wards"),
            countDistinct("Community Area").alias("num_community_areas"),
            countDistinct("Location Description").alias("num_location_descriptions")
        )
    aggregated_df = aggregated_df.withColumn("total_crimes", col("total_violent_crimes") + col("total_non_violent_crimes"))

    # Create lagged features
    window_spec = Window.partitionBy('Beat').orderBy('year', 'week_of_year')
    for num_weeks_lag in [1, 2, 3, 4]:
        aggregated_df = aggregated_df.withColumn(f'total_crimes_lag_{num_weeks_lag}', lag('total_crimes', num_weeks_lag).over(window_spec))
        aggregated_df = aggregated_df.withColumn(f'total_arrests_lag_{num_weeks_lag}', lag('total_arrests', num_weeks_lag).over(window_spec))
        aggregated_df = aggregated_df.withColumn(f'total_domestic_crimes_lag_{num_weeks_lag}', lag('total_domestic_crimes', num_weeks_lag).over(window_spec))

    # Fill missing values with 0
    aggregated_df = aggregated_df.na.fill(0)

    return aggregated_df

if __name__ == "__main__":
    # Start Spark session
    spark = initialize_spark_session()
    
    # Load datasets
    crime_df = load_crime_data(spark)
    iucr_df = load_iucr_data(spark)
    pop_df = load_pop_data(spark)
    race_df = load_race_data(spark)
    race_df.show()
    fs_df = load_fs_data(spark)

    # Perform feature engineering
    feature_engineered_df = feature_engineering(crime_df, iucr_df)
