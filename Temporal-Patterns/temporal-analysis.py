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

def filter_and_group(data_frame, column):
    """Filter arrests data and group by specified column."""
    arrests_data = data_frame.filter(data_frame['Arrest'] == 'true')
    return arrests_data.groupBy(column).count().orderBy(column)

def plot_data(data_frame, x_column, y_column, title, ax):
    """Plot data on the given axis."""
    data_frame.plot(kind='bar', x=x_column, y=y_column, ax=ax, title=title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

def save_combined_data(hourly_df, weekly_df, monthly_df, file_name):
    """Save combined data to a single text file."""
    with open(file_name, 'w') as f:
        f.write("Hourly Arrests\n")
        f.write(hourly_df.to_string(index=False))
        f.write("\n\nWeekly Arrests\n")
        f.write(weekly_df.to_string(index=False))
        f.write("\n\nMonthly Arrests\n")
        f.write(monthly_df.to_string(index=False))

def main():
    # Load and process the crime data
    crime_data_path = "hdfs://wolf:9000/user/uhw4967/crime_data/Crimes_-_2001_to_present.csv"
    crime_data = load_data(crime_data_path)
    processed_data = process_data(crime_data)

    # Aggregate data by hour, day of week, and month
    hourly_data = filter_and_group(processed_data, 'Hour')
    weekly_data = filter_and_group(processed_data, 'DayOfWeek')
    monthly_data = filter_and_group(processed_data, 'Month')
    
    # Convert Spark DataFrames to pandas DataFrames for plotting and saving
    hourly_data_pd = hourly_data.toPandas()
    weekly_data_pd = weekly_data.toPandas()
    monthly_data_pd = monthly_data.toPandas()

    # Plot and save all plots in a single PNG file
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plot_data(hourly_data_pd, 'Hour', 'count', 'Hourly Arrests', axs[0])
    plot_data(weekly_data_pd, 'DayOfWeek', 'count', 'Weekly Arrests', axs[1])
    plot_data(monthly_data_pd, 'Month', 'count', 'Monthly Arrests', axs[2])
    plt.tight_layout()
    plt.savefig('kuemmerle_q4.png')
    
    # Save combined data to a single text file
    save_combined_data(hourly_data_pd, weekly_data_pd, monthly_data_pd, 'kuemmerle_q4.txt')


if __name__ == "__main__":
    main()
