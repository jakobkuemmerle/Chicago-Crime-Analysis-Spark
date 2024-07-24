# Import necessary libraries
from pyspark import SparkContext
from pyspark.mllib.stat import Statistics
from pyspark.mllib.regression import LabeledPoint
from operator import add
from datetime import datetime
import re

# Initialize SparkContext
sc = SparkContext()

# Load crime data RDD
crime_data_rdd = sc.textFile("hdfs://wolf:9000/user/uhw4967/crime_data/Crimes_-_2001_to_present.csv")
header = crime_data_rdd.first()
crime_data = crime_data_rdd.filter(lambda line: line != header)

# Find the current year
current_year = crime_data.map(lambda x: int(x.split(",")[2].split("/")[2].split(" ")[0])).max()

# Filter recent crime data for the last 3 years
recent_crime = crime_data.filter(lambda x: int(x.split(",")[2].split("/")[2].split(" ")[0]) > current_year - 3)

# Count crime events per block
block_counts = recent_crime.map(lambda x: x.split(",")).map(lambda x: (x[3],1)).reduceByKey(add)

# Write top 10 blocks with the highest crime events in the last 3 years to file
with open('kuemmerle_q2.txt', 'w') as file:
    file.write("(1) Top 10 blocks in crime events in the last 3 years:\n")
    for item in block_counts.sortBy(lambda x: x[1], ascending=False).take(10):
        file.write(str(item) + '\n')

# Filter recent crime data for the last 5 years
recent_crime_5 = crime_data.filter(lambda x: int(x.split(",")[2].split("/")[2].split(" ")[0]) > current_year - 5)

# Function to remove commas within quotes
def remove_commas_within_quotes(line):
    pattern = r'"([^"]*)"'
    def remove_commas(match):
        return match.group(0).replace(',', '')
    return re.sub(pattern, remove_commas, line)

# Clean recent crime data
recent_crime_clean = recent_crime_5.map(remove_commas_within_quotes).map(lambda x: x.split(","))

# Map crime data to (beat, year) pairs and reduce by key to count occurrences
beat_year_counts = recent_crime_clean.map(lambda x: ((x[10], x[2].split("/")[2].split(" ")[0]), 1)).reduceByKey(add)

# Sort beats by key and filter those with data for all 5 years
beat_counts = beat_year_counts.sortByKey().map(lambda x: (x[0][0], x[1]))
beat_counts_dic = beat_counts.groupByKey().mapValues(list).filter(lambda x: len(x[1]) == 5).sortByKey()

# Extract keys and values
beat_counts_keys = beat_counts_dic.map(lambda x: x[0]).collect()
beat_counts_values = beat_counts_dic.map(lambda x: x[1])

# Create RDD of beats with index and map beat counts to index
beats_rdd = beat_counts_dic.values().zipWithIndex().map(lambda x: (x[1], x[0]))
beats_map = beats_rdd.flatMap(lambda beat: [(index, beat[0], value) for index, value in enumerate(beat[1])])

# Group beat values by index and sort
beats_map_list = beats_map.groupBy(lambda x: x[0]).map(lambda x: sorted(x[1]))

# Extract beat lists
beats_list = beats_map_list.map(lambda sublist: [item[2] for item in sublist])

# Calculate correlation matrix
correlation_matrix = Statistics.corr(beats_list)

# Flatten and zip correlation matrix
correlation_rdd = sc.parallelize(correlation_matrix)
correlation_values_upper = correlation_rdd.zipWithIndex().flatMap(
    lambda row_with_index: [((row_with_index[1], idx + row_with_index[1] + 1), value) for idx, value in enumerate(row_with_index[0][row_with_index[1]+1:])])

# Find beat pairs with highest correlation
max_correlation_indices = correlation_values_upper.sortBy(lambda x: -x[1]).take(20)
beat_pairs = []
for idx in max_correlation_indices:
    beat_pairs.append((beat_counts_keys[idx[0][0]], beat_counts_keys[idx[0][1]], idx[1]))

# Write highest correlating beat pairs to file
with open('kuemmerle_q2.txt', 'a') as file:
    file.write("\n(2) Highest Correlating Beats:\n")
    for i in beat_pairs:
        file.write(str(i) + '\n')

# Define start and end dates for Daley and Emanuel administrations
daley_start_date = datetime.strptime("04/24/1989", "%m/%d/%Y")
daley_end_date = datetime.strptime("05/16/2011", "%m/%d/%Y")
emanuel_start_date = datetime.strptime("05/16/2011", "%m/%d/%Y")
emanuel_end_date = datetime.strptime("05/20/2019", "%m/%d/%Y")

# Filter crime data for Daley and Emanuel administrations
filter_daley = crime_data.filter(lambda line: (daley_start_date <= datetime.strptime(line.split(",")[2].split(" ")[0], "%m/%d/%Y") < daley_end_date))
filter_emanuel = crime_data.filter(lambda line: (emanuel_start_date <= datetime.strptime(line.split(",")[2].split(" ")[0], "%m/%d/%Y") < emanuel_end_date))

# Count crimes by beat for Daley and Emanuel administrations
delay_count = filter_daley.map(remove_commas_within_quotes).map(lambda x: (x.split(",")[10], 1)).reduceByKey(add)
emanuel_count = filter_emanuel.map(remove_commas_within_quotes).map(lambda x: (x.split(",")[10], 1)).reduceByKey(add)

# Calculate average daily crime rates for Daley and Emanuel administrations
data_start = datetime(2001, 1, 1)
delay_count_avg = delay_count.map(lambda x: (x[0], x[1] / (daley_end_date - data_start).days))
emanuel_count_avg = emanuel_count.map(lambda x: (x[0], x[1] / (emanuel_end_date - emanuel_start_date).days))

# Join counts for Daley and Emanuel administrations
joint_count = delay_count_avg.join(emanuel_count_avg)

# Extract counts for Daley and Emanuel administrations
delay_full = joint_count.map(lambda line: line[1][0])
emanuel_full = joint_count.map(lambda line: line[1][1])

# Create LabeledPoint RDDs for Daley and Emanuel administrations
delay_labeled_points = delay_full.map(lambda x: LabeledPoint(0, [x])) 
emanuel_labeled_points = emanuel_full.map(lambda x: LabeledPoint(1, [x])) 

# Union LabeledPoint RDDs
all_labeled_points = delay_labeled_points.union(emanuel_labeled_points)

# Perform Chi-Squared test to evaluate differences in Mayors
chi_squared_result = Statistics.chiSqTest(all_labeled_points)[0]

# Write Chi-Squared test result to file
with open('kuemmerle_q2.txt', 'a') as file:
    file.write("\nChi-Square Test Result to evaluate differences in Mayors:\n")
    file.write(str(chi_squared_result.pValue) + '\n')

# Stop SparkContext
sc.stop()
