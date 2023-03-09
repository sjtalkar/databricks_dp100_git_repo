# Databricks notebook source
# MAGIC %md
# MAGIC # Working with data in Azure Databricks
# MAGIC 
# MAGIC **Technical Accomplishments:**
# MAGIC - viewing available tables
# MAGIC - loading table data in dataframes
# MAGIC - loading file/dbfs data in dataframes
# MAGIC - using spark for simple queries
# MAGIC - using spark to show the data and its structure
# MAGIC - using spark for complex queries
# MAGIC - using Databricks' `display` for custom visualisations

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Attach notebook to your cluster
# MAGIC Before executing any cells in the notebook, you need to attach it to your cluster. Make sure that the cluster is running.
# MAGIC 
# MAGIC In the notebook's toolbar, select the drop down arrow next to Detached, and then select your cluster under Attach to.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## About Spark DataFrames
# MAGIC 
# MAGIC Spark DataFrames are distributed collections of data, organized into rows and columns, similar to traditional SQL tables.
# MAGIC 
# MAGIC A DataFrame can be operated on using relational transformations, through the Spark SQL API, which is available in Scala, Java, Python, and R.
# MAGIC 
# MAGIC We will use Python in our notebook. 
# MAGIC 
# MAGIC We often refer to DataFrame variables using `df`.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Loading data into dataframes

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### View available data
# MAGIC 
# MAGIC To check the data available in our Databricks environment we can use the `%sql` magic and query our tables:

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from nyc_taxi;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Reading data from our tables
# MAGIC 
# MAGIC Using Spark, we can read data into dataframes. 
# MAGIC 
# MAGIC It is important to note that spark has read/write support for a widely set of formats. 
# MAGIC It can use
# MAGIC * csv
# MAGIC * json
# MAGIC * parquet
# MAGIC * orc
# MAGIC * avro
# MAGIC * hive tables
# MAGIC * jdbc
# MAGIC 
# MAGIC We can read our data from the tables (since we already imported the initial csv as Databricks tables).

# COMMAND ----------

df = spark.sql("SELECT * FROM nyc_taxi")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Reading data from the DBFS
# MAGIC 
# MAGIC We can also read the data from the original files we've uploaded; or indeed from any other file available in the DBFS. 
# MAGIC 
# MAGIC The code is the same regardless of whether a file is local or in mounted remote storage that was mounted, thanks to DBFS mountpoints

# COMMAND ----------

df = spark.read.csv('dbfs:/FileStore/tables/nyc_taxi.csv', header=True, inferSchema=True)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### DataFrame size
# MAGIC 
# MAGIC Use `count` to determine how many rows of data we have in a dataframe.

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### DataFrame structure
# MAGIC 
# MAGIC To get information about the schema associated with our dataframe we can use `printSchema`:

# COMMAND ----------

df.printSchema

# COMMAND ----------

# MAGIC %md
# MAGIC ### show(..) vs display(..)
# MAGIC * `show(..)` is part of core spark - `display(..)` is specific to our notebooks.
# MAGIC * `show(..)` has parameters for truncating both columns and rows - `display(..)` does not.
# MAGIC * `show(..)` is a function of the `DataFrame`/`Dataset` class - `display(..)` works with a number of different objects.
# MAGIC * `display(..)` is more powerful - with it, you can...
# MAGIC   * Download the results as CSV
# MAGIC   * Render line charts, bar chart & other graphs, maps and more.
# MAGIC   * See up to 1000 records at a time.
# MAGIC   
# MAGIC For the most part, the difference between the two is going to come down to preference.
# MAGIC 
# MAGIC Remember, the `display` function is Databricks specific. It is not available in standard spark code.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Querying dataframes
# MAGIC 
# MAGIC Once that spark has the data, we can manipulate it using spark SQL API.
# MAGIC 
# MAGIC We can easily use the spark SQL dsl to do joins, aggregations, filtering. 
# MAGIC We can change the data structure, add or drop columns, or change the column types.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We will use the python function we've already defined to convert Celsius degrees to Fahrenheit degrees.

# COMMAND ----------

def celsiusToFahrenheit(source_temp=None):
    return(source_temp * (9.0/5.0)) + 32.0
  
celsiusToFahrenheit(27)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We will adapt it as a udf (user defined function) to make it usable with Spark's dataframes API.
# MAGIC 
# MAGIC And we will use it to enrich our source data.

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import *

udfCelsiusToFahrenheit = udf(lambda z: celsiusToFahrenheit(z), DoubleType())

display(df.filter(col('temperature').isNotNull()) \
  .withColumn("tempC", col("temperature").cast(DoubleType())) \
  .select(col("tempC"), udfCelsiusToFahrenheit(col("tempC")).alias("tempF")))
  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC More complex SQL functions are available in spark: 
# MAGIC 
# MAGIC * grouping, sorting, limits, count
# MAGIC * aggregations: agg, max, sum
# MAGIC * windowing: partitionBy, count over, max over
# MAGIC 
# MAGIC For example may want to add a row-number column to our source data. Window functions will help with such complex queries:

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id

display(df.orderBy('tripDistance', ascending=False) \
  .withColumn('rowno', row_number().over(Window.orderBy(monotonically_increasing_id()))))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Data cleaning
# MAGIC 
# MAGIC Before using the source data, we have to validate the contents. Let's see if there are any duplicates:

# COMMAND ----------

df.count() - df.dropDuplicates().count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Some columns might be missing. We check the presence of null values for each column.

# COMMAND ----------

display(df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Since some of our columns seem to have such null values, we'll have to fix these rows.
# MAGIC 
# MAGIC We could either replace null values using `fillna` or ignore such rows using `dropna`

# COMMAND ----------

df = df.fillna({'passengerCount':'1'}).dropna()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Explore Summary Statistics and Data Distribution
# MAGIC Predictive modeling is based on statistics and probability, so we should take a look at the summary statistics for the columns in our data. The **describe** function returns a dataframe containing the **count**, **mean**, **standard deviation**, **minimum**, and **maximum** values for each numeric column.

# COMMAND ----------

display(df.describe())


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Visualizing data
# MAGIC 
# MAGIC Azure Databricks has custom support for displaying data. 
# MAGIC 
# MAGIC The `display(..)` command has multiple capabilities:
# MAGIC * Presents up to 1000 records.
# MAGIC * Exporting data as CSV.
# MAGIC * Rendering a multitude of different graphs.
# MAGIC * Rendering geo-located data on a world map.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Let's take a look at our data using databricks visualizations:
# MAGIC * Run the cell below
# MAGIC * click on the second icon underneath the executed cell and choose `Bar`
# MAGIC * click on the `Plot Options` button to configure the graph
# MAGIC   * drag the `tripDistance` into the `Keys` list
# MAGIC   * drag the `totalAmount` into the `Values` list
# MAGIC   * choose `Aggregation` as `AVG`
# MAGIC   * click `Apply`

# COMMAND ----------

dfClean = df.select(col("tripDistance"), col("totalAmount")).dropna()

display(dfClean)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Note that the points form a diagonal line, which indicates a strong linear relationship between the trip distance and the total amount. This linear relationship shows a correlation between these two values, which we can measure statistically. 
# MAGIC 
# MAGIC The `corr` function calculates a correlation value between -1 and 1, indicating the strength of correlation between two fields. A strong positive correlation (near 1) indicates that high values for one column are often found with high values for the other, which a strong negative correlation (near -1) indicates that low values for one column are often found with high values for the other. A correlation near 0 indicates little apparent relationship between the fields.

# COMMAND ----------

dfClean.corr('tripDistance', 'totalAmount')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Predictive modeling is largely based on statistical relationships between fields in the data. To design a good model, you need to understand how the data points relate to one another.
# MAGIC 
# MAGIC A common way to start exploring relationships is to create visualizations that compare two or more data values. For example, modify the Plot Options of the chart above to compare the arrival delays for each carrier:
# MAGIC 
# MAGIC * Keys: temperature
# MAGIC * Series Groupings: month_num
# MAGIC * Values: snowDeprh
# MAGIC * Aggregation: avg
# MAGIC * Display Type: Line Chart

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The plot now shows the relation between the month, the snow amount and the recorded temperature.
