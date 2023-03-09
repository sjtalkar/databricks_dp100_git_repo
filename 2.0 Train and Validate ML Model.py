# Databricks notebook source
# MAGIC %md
# MAGIC # Training and Validating a Machine Learning Model
# MAGIC 
# MAGIC Linear regression is the most commonly employed machine learning model since it is highly interpretable and well studied.  This is often the first pass for data scientists modeling continuous variables.  This notebook trains a multivariate regression model and interprets the results. This notebook is organized in two sections:
# MAGIC 
# MAGIC - Exercise 1: Training a Model
# MAGIC - Exercise 2: Validating a Model

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to load common libraries.

# COMMAND ----------

import urllib.request
import os
import numpy as np
from pyspark.sql.types import * 
from pyspark.sql.functions import col, lit
from pyspark.sql.functions import udf
import matplotlib
import matplotlib.pyplot as plt
print("Imported common libraries.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the training data
# MAGIC 
# MAGIC In this notebook, we will be using a subset of NYC Taxi & Limousine Commission - green taxi trip records available from [Azure Open Datasets]( https://azure.microsoft.com/en-us/services/open-datasets/). The data is enriched with holiday and weather data. Each row of the table represents a taxi ride that includes columns such as number of passengers, trip distance, datetime information, holiday and weather information, and the taxi fare for the trip.
# MAGIC 
# MAGIC Run the following cell to load the table into a Spark dataframe and reivew the dataframe.

# COMMAND ----------

dataset = spark.sql("select * from nyc_taxi")
display(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 1: Training a Model
# MAGIC 
# MAGIC In this section we will use the Spark's machine learning library, `MLlib` to train a `NYC Taxi Fare Predictor` machine learning model. We will train a multivariate regression model to predict taxi fares in New York City based on input features such as, number of passengers, trip distance, datetime, holiday information and weather information. Before we start, let's review the three main abstractions that are provided in the `MLlib`:<br><br>
# MAGIC 
# MAGIC 1. A **transformer** takes a DataFrame as an input and returns a new DataFrame with one or more columns appended to it.  
# MAGIC   - Transformers implement a `.transform()` method.  
# MAGIC 2. An **estimator** takes a DataFrame as an input and returns a model, which itself is a transformer.
# MAGIC   - Estimators implements a `.fit()` method.
# MAGIC 3. A **pipeline** combines together transformers and estimators to make it easier to combine multiple algorithms.
# MAGIC   - Pipelines implement a `.fit()` method.
# MAGIC   
# MAGIC These basic building blocks form the machine learning process in Spark from featurization through model training and deployment.

# COMMAND ----------

# MAGIC %md
# MAGIC -sandbox
# MAGIC 
# MAGIC ### Featurization of the training data
# MAGIC 
# MAGIC Machine learning models are only as strong as the data they see and can only work on numerical data.  **Featurization is the process of creating this input data for a model.** In this section we will build derived features and create a pipeline of featurization steps.

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to engineer the cyclical features to represent `hour_of_day`. Also, we will drop rows with null values in the `totalAmount` column and convert the column ` isPaidTimeOff ` as integer type.

# COMMAND ----------

def get_sin_cosine(value, max_value):
  sine =  np.sin(value * (2.*np.pi/max_value))
  cosine = np.cos(value * (2.*np.pi/max_value))
  return (sine.tolist(), cosine.tolist())

schema = StructType([
    StructField("sine", DoubleType(), False),
    StructField("cosine", DoubleType(), False)
])

get_sin_cosineUDF = udf(get_sin_cosine, schema)

dataset = dataset.withColumn("udfResult", get_sin_cosineUDF(col("hour_of_day"), lit(24))).withColumn("hour_sine", col("udfResult.sine")).withColumn("hour_cosine", col("udfResult.cosine")).drop("udfResult").drop("hour_of_day")

dataset = dataset.filter(dataset.totalAmount.isNotNull())

dataset = dataset.withColumn("isPaidTimeOff", col("isPaidTimeOff").cast("integer"))

display(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to create stages in our featurization pipeline to scale the numerical features and to encode the categorical features.

# COMMAND ----------

from pyspark.ml.feature import Imputer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline

numerical_cols = ["passengerCount", "tripDistance", "snowDepth", "precipTime", "precipDepth", "temperature", "hour_sine", "hour_cosine"]
categorical_cols = ["day_of_week", "month_num", "normalizeHolidayName", "isPaidTimeOff"]
label_column = "totalAmount"

stages = []

inputCols = ["passengerCount"]
outputCols = ["passengerCount"]
imputer = Imputer(strategy="median", inputCols=inputCols, outputCols=outputCols)
stages += [imputer]

assembler = VectorAssembler().setInputCols(numerical_cols).setOutputCol('numerical_features')
scaler = MinMaxScaler(inputCol=assembler.getOutputCol(), outputCol="scaled_numerical_features")
stages += [assembler, scaler]

for categorical_col in categorical_cols:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categorical_col, outputCol=categorical_col + "_index", handleInvalid="skip")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categorical_col + "_classVector"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]
    
print("Created stages in our featurization pipeline to scale the numerical features and to encode the categorical features.")

# COMMAND ----------

# MAGIC %md
# MAGIC Use a `VectorAssembler` to combine all the feature columns into a single vector column named **features**.

# COMMAND ----------

assemblerInputs = [c + "_classVector" for c in categorical_cols] + ["scaled_numerical_features"]
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
print("Used a VectorAssembler to combine all the feature columns into a single vector column named features.")

# COMMAND ----------

# MAGIC %md
# MAGIC -sandbox
# MAGIC **Run the stages as a Pipeline**
# MAGIC 
# MAGIC The pipeline is itself is now an `estimator`.  Call the pipeline's `fit` method and then `transform` the original dataset. This puts the data through all of the feature transformations we described in a single call. Observe the new columns, especially column: **features**.

# COMMAND ----------

partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(dataset)
preppedDataDF = pipelineModel.transform(dataset)

display(preppedDataDF)

# COMMAND ----------

# MAGIC %md
# MAGIC -sandbox
# MAGIC 
# MAGIC ### Train a multivariate regression model
# MAGIC 
# MAGIC A multivariate regression takes an arbitrary number of input features. The equation for multivariate regression looks like the following where each feature `p` has its own coefficient:
# MAGIC 
# MAGIC &nbsp;&nbsp;&nbsp;&nbsp;`Y ≈ β<sub>0</sub> + β<sub>1</sub>X<sub>1</sub> + β<sub>2</sub>X<sub>2</sub> + ... + β<sub>p</sub>X<sub>p</sub>`

# COMMAND ----------

# MAGIC %md
# MAGIC Split the featurized training data for training and validating the model

# COMMAND ----------

(trainingData, testData) = preppedDataDF.randomSplit([0.7, 0.3], seed=97)
print("The training data is split for training and validating the model: 70-30 split.")

# COMMAND ----------

# MAGIC %md
# MAGIC Create the estimator `LinearRegression` and call its `fit` method to get back the trained ML model (`lrModel`). You can read more about [Linear Regression] from the [classification and regression] section of MLlib Programming Guide.
# MAGIC 
# MAGIC [classification and regression]: https://spark.apache.org/docs/latest/ml-classification-regression.html
# MAGIC [Linear Regression]: https://spark.apache.org/docs/3.1.1/ml-classification-regression.html#linear-regression

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol=label_column)

lrModel = lr.fit(trainingData)

print(lrModel)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 2: Validating a Model

# COMMAND ----------

# MAGIC %md
# MAGIC -sandbox
# MAGIC 
# MAGIC From the trained model summary, let’s review some of the model performance metrics such as, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R<sup>2</sup> score. We will also look at the multivariate model’s coefficients.

# COMMAND ----------

summary = lrModel.summary
print("RMSE score: {} \nMAE score: {} \nR2 score: {}".format(summary.rootMeanSquaredError, summary.meanAbsoluteError, lrModel.summary.r2))
print("")
print("β0 (intercept): {}".format(lrModel.intercept))
i = 0
for coef in lrModel.coefficients:
  i += 1
  print("β{} (coefficient): {}".format(i, coef))

# COMMAND ----------

# MAGIC %md
# MAGIC -sandbox
# MAGIC 
# MAGIC Evaluate the model performance using the hold-back  dataset. Observe that the RMSE and R<sup>2</sup> score on holdback dataset is slightly degraded compared to the training summary. A big disparity in performance metrics between training and hold-back dataset can be an indication of model overfitting the training data.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

predictions = lrModel.transform(testData)
evaluator = RegressionEvaluator(
    labelCol=label_column, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
evaluator = RegressionEvaluator(
    labelCol=label_column, predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(predictions)
print("MAE on test data = %g" % mae)
evaluator = RegressionEvaluator(
    labelCol=label_column, predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print("R2 on test data = %g" % r2)

# COMMAND ----------

# MAGIC %md
# MAGIC **Compare the summary statistics between the true values and the model predictions**

# COMMAND ----------

display(predictions.select(["totalAmount",  "prediction"]).describe())

# COMMAND ----------

# MAGIC %md
# MAGIC **Visualize the plot between true values and the model predictions**

# COMMAND ----------

p_df = predictions.select(["totalAmount",  "prediction"]).toPandas()
true_value = p_df.totalAmount
predicted_value = p_df.prediction

plt.figure(figsize=(10,10))
plt.scatter(true_value, predicted_value, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(predicted_value), max(true_value))
p2 = min(min(predicted_value), min(true_value))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()
