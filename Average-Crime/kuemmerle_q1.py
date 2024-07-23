from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col, month, year, sum as sql_sum, countDistinct
import matplotlib.pyplot as plt

def start_spark_session(app_name):
    """Start a Spark session with the specified application name."""
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def read_crime_data(spark, file_path):
    """Read crime data from a CSV file located at the given path."""
    return spark.read.csv(file_path, header=True)

def transform_data(crime_df):
    """Transform the crime data by parsing dates and extracting month and year."""
    transformed_df = crime_df.withColumn("Date", to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a"))
    transformed_df = transformed_df.withColumn("Month", month("Date"))
    transformed_df = transformed_df.withColumn("Year", year("Date"))
    return transformed_df

def compute_average_monthly_crimes(transformed_df):
    """Compute the average number of crimes per month over all years."""
    monthly_counts = transformed_df.groupBy("Month", "Year").count().withColumnRenamed("count", "MonthlyCount")
    avg_monthly_crimes = monthly_counts.groupBy("Month").agg(
        (sql_sum("MonthlyCount") / countDistinct("Year")).alias("AverageCrimes")
    )
    return avg_monthly_crimes

def plot_average_crimes(average_crimes):
    """Plot the average crime events per month using matplotlib and save as an image."""
    sorted_data = sorted(average_crimes, key=lambda x: x[0])  # Sort by month
    months = [x[0] for x in sorted_data]
    avg_crimes = [x[1] for x in sorted_data]

    plt.figure(figsize=(10, 5))
    plt.bar(months, avg_crimes, color='blue')
    plt.xlabel('Month')
    plt.ylabel('Average Crime Events')
    plt.title('Average Crime Events by Month')
    plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.savefig('kuemmerle_q1.png')

def main():
    """Main function to run the Spark application for analyzing Chicago crime data."""
    app_name = "Chicago_Crime_Analysis"
    crime_data_path = 'hdfs://wolf:9000/user/uhw4967/crime_data/Crimes_-_2001_to_present.csv'

    # Start Spark session
    spark = start_spark_session(app_name)

    # Read and display the crime data
    crime_df = read_crime_data(spark, crime_data_path)
    crime_df.printSchema()
    crime_df.show(5, truncate=False)

    # Transform and display the cleaned data
    cleaned_crime_df = transform_data(crime_df)
    cleaned_crime_df.select('Date', 'Month', 'Year').show(5, truncate=False)

    # Compute and collect the average monthly crimes
    avg_monthly_crimes_df = compute_average_monthly_crimes(cleaned_crime_df)
    avg_monthly_crimes_result = avg_monthly_crimes_df.collect()

    # Plot and save the average crimes per month
    plot_average_crimes(avg_monthly_crimes_result)

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
