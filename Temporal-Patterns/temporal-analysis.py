from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import to_timestamp, hour, dayofweek, month
import matplotlib.pyplot as plt
import pandas as pd

# Set up Spark configuration and context
conf = SparkConf().setAppName("Chicago Crime Data Analysis")
spark_context = SparkContext(conf=conf)
spark_context.setLogLevel("ERROR")
sql_context = SQLContext(spark_context)

def load_data(file_path):
    """Load crime data from a CSV file located in HDFS."""
    data = sql_context.read.format("csv").option("header", "true").load(file_path)
    return data.withColumn('Timestamp', to_timestamp('Date', 'MM/dd/yyyy hh:mm:ss a'))

def process_data(crime_data):
    """Extract hour, day of the week, and month from the timestamp."""
    crime_data = crime_data.withColumn('Hour', hour(crime_data['Timestamp']))
    crime_data = crime_data.withColumn('DayOfWeek', dayofweek(crime_data['Timestamp']))
    return crime_data.withColumn('Month', month(crime_data['Timestamp']))

def main():
    # Load and process the crime data
    crime_data_path = "hdfs://wolf:9000/user/uhw4967/crime_data/Crimes_-_2001_to_present.csv"
    crime_data = load_data(crime_data_path)
    processed_data = process_data(crime_data)


if __name__ == "__main__":
    main()
