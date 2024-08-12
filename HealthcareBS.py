# Databricks notebook source
# Load the dataset
file_path = "/FileStore/Blood_samples_dataset_balanced_2_f_.csv"
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Display the dataset
data.display(5)


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("BloodSampleClassification").getOrCreate()

# Correct file path
file_path = "/FileStore/Blood_samples_dataset_balanced_2_f_.csv"

# Load the dataset
data = spark.read.csv(file_path, header=True, inferSchema=True)
data.display(10)  # Show the first few rows
data.printSchema()  # Print schema to verify data types

# Check distinct values in the 'Disease' column
data.select("Disease").distinct().display()

# Filter dataset for binary classification if it has more than two distinct values
binary_classes = ['Class1', 'Class2']  # Replace with actual class names if different
data = data.filter(data.Disease.isin(binary_classes))

# Print counts after filtering
print(f"Data Count After Filtering: {data.count()}")

# Handling missing values
data = data.dropna()

# Print counts after dropping NA
print(f"Data Count After Dropping NA: {data.count()}")

# Encode the label column 'Disease' to numeric
indexer = StringIndexer(inputCol="Disease", outputCol="label")
data = indexer.fit(data).transform(data)

# Define feature columns (excluding the newly created 'label' column)
feature_columns = [col for col in data.columns if col not in ['Disease', 'label']]
print(f"Feature Columns: {feature_columns}")

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# Select only the features and label columns
data = data.select("features", "label")

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

# Check the counts of the train and test datasets
print(f"Train Data Count: {train_data.count()}")
print(f"Test Data Count: {test_data.count()}")

if train_data.count() > 0 and test_data.count() > 0:
    # Train a Logistic Regression model
    lr = LogisticRegression(labelCol="label", featuresCol="features")
    model = lr.fit(train_data)

    # Make predictions on the test data
    predictions = model.transform(test_data)
    predictions.select("features", "label", "prediction").show(5)

    # Ensure the predictions DataFrame contains the correct columns and is not empty
    required_columns = ["label", "prediction", "rawPrediction"]
    missing_columns = [col for col in required_columns if col not in predictions.columns]
    if predictions.count() > 0 and not missing_columns:
        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        accuracy = evaluator.evaluate(predictions)
        print(f"Accuracy: {accuracy}")
    else:
        if missing_columns:
            print(f"Missing columns in predictions DataFrame: {missing_columns}")
        else:
            print("Predictions DataFrame is empty. Cannot evaluate the model.")

    # Save the predictions to a Delta table
    predictions.write.format("delta").mode("overwrite").save("/mnt/delta/blood_sample_predictions")

    # Optionally, create a table from the Delta file
    spark.sql("CREATE TABLE IF NOT EXISTS blood_sample_predictions USING DELTA LOCATION '/mnt/delta/blood_sample_predictions'")
else:
    print("Train or Test Data is empty. Model training and evaluation cannot proceed.")


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("BloodSampleClassification").getOrCreate()

# Correct file path
file_path = "/FileStore/Blood_samples_dataset_balanced_2_f_.csv"

# Load the dataset
data = spark.read.csv(file_path, header=True, inferSchema=True)
data.(5)  # Show the first few rows
data.printSchema()  # Print schema to verify data types

# Check distinct values in the 'Disease' column
data.select("Disease").distinct().show()

# Filter dataset for binary classification
binary_classes = ['Positive', 'Negative']  # Replace with actual class names
data = data.filter(data.Disease.isin(binary_classes))

# Print counts after filtering
print(f"Data Count After Filtering: {data.count()}")

# Handling missing values
data = data.dropna()
print(f"Data Count After Dropping NA: {data.count()}")

# Encode the label column 'Disease' to numeric
indexer = StringIndexer(inputCol="Disease", outputCol="label")
data = indexer.fit(data).transform(data)

# Define feature columns
feature_columns = [col for col in data.columns if col not in ['Disease', 'label']]
print(f"Feature Columns: {feature_columns}")

# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# Select only the features and label columns
data = data.select("features", "label")

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

# Check the counts of the train and test datasets
print(f"Train Data Count: {train_data.count()}")
print(f"Test Data Count: {test_data.count()}")

if train_data.count() > 0 and test_data.count() > 0:
    # Train a Logistic Regression model
    lr = LogisticRegression(labelCol="label", featuresCol="features")
    model = lr.fit(train_data)

    # Make predictions on the test data
    predictions = model.transform(test_data)
    predictions.select("features", "label", "prediction").show(5)

    # Ensure the predictions DataFrame contains the correct columns and is not empty
    required_columns = ["label", "prediction", "rawPrediction"]
    missing_columns = [col for col in required_columns if col not in predictions.columns]
    if predictions.count() > 0 and not missing_columns:
        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        accuracy = evaluator.evaluate(predictions)
        print(f"Accuracy: {accuracy}")
    else:
        if missing_columns:
            print(f"Missing columns in predictions DataFrame: {missing_columns}")
        else:
            print("Predictions DataFrame is empty. Cannot evaluate the model.")

    # Save the predictions to a Delta table
    predictions.write.format("delta").mode("overwrite").save("/mnt/delta/blood_sample_predictions")

    # Optionally, create a table from the Delta file
    spark.sql("CREATE TABLE IF NOT EXISTS blood_sample_predictions USING DELTA LOCATION '/mnt/delta/blood_sample_predictions'")
else:
    print("Train or Test Data is empty. Model training and evaluation cannot proceed.")


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("BloodSampleClassification").getOrCreate()

# Correct file path
file_path = "/FileStore/Blood_samples_dataset_balanced_2_f_.csv"

# Load the dataset
data = spark.read.csv(file_path, header=True, inferSchema=True)
data.show(5)  # Show the first 5 rows
data.printSchema()  # Print schema to verify data types

# Check distinct values in the 'Disease' column
data.select("Disease").distinct().show()

# Filter dataset for binary classification
binary_classes = ['Positive', 'Negative']  # Replace with actual class names
data = data.filter(data.Disease.isin(binary_classes))

# Print counts after filtering
print(f"Data Count After Filtering: {data.count()}")

# Handling missing values
data = data.dropna()
print(f"Data Count After Dropping NA: {data.count()}")

# Encode the label column 'Disease' to numeric
indexer = StringIndexer(inputCol="Disease", outputCol="label")
data = indexer.fit(data).transform(data)

# Define feature columns
feature_columns = [col for col in data.columns if col not in ['Disease', 'label']]
print(f"Feature Columns: {feature_columns}")

# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# Select only the features and label columns
data = data.select("features", "label")

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

# Check the counts of the train and test datasets
print(f"Train Data Count: {train_data.count()}")
print(f"Test Data Count: {test_data.count()}")

if train_data.count() > 0 and test_data.count() > 0:
    # Train a Logistic Regression model
    lr = LogisticRegression(labelCol="label", featuresCol="features")
    model = lr.fit(train_data)

    # Save the trained model
    model.save("/FileStore/models/logistic_regression_blood_sample")

    # Make predictions on the test data
    predictions = model.transform(test_data)
    predictions.select("features", "label", "prediction").show(5)

    # Ensure the predictions DataFrame contains the correct columns and is not empty
    required_columns = ["label", "prediction", "rawPrediction"]
    missing_columns = [col for col in required_columns if col not in predictions.columns]
    if predictions.count() > 0 and not missing_columns:
        # Evaluate the model using different metrics
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        accuracy = evaluator.evaluate(predictions)
        print(f"Accuracy: {accuracy}")

        # Precision, Recall, F1-Score
        precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel")
        recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel")
        f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

        precision = precision_evaluator.evaluate(predictions)
        recall = recall_evaluator.evaluate(predictions)
        f1_score = f1_evaluator.evaluate(predictions)

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1_score}")
    else:
        if missing_columns:
            print(f"Missing columns in predictions DataFrame: {missing_columns}")
        else:
            print("Predictions DataFrame is empty. Cannot evaluate the model.")

    # Save the predictions to a Delta table
    predictions.write.format("delta").mode("overwrite").save("/mnt/delta/blood_sample_predictions")

    # Optionally, create a table from the Delta file
    spark.sql("CREATE TABLE IF NOT EXISTS blood_sample_predictions USING DELTA LOCATION '/mnt/delta/blood_sample_predictions'")

    # Interactive Input Handling for Predictions
    def make_prediction(input_data):
        # Convert input data to Spark DataFrame
        input_df = spark.createDataFrame([input_data], feature_columns)
        input_data_transformed = assembler.transform(input_df)

        # Load the saved model
        loaded_model = LogisticRegressionModel.load("/FileStore/models/logistic_regression_blood_sample")

        # Make predictions
        prediction = loaded_model.transform(input_data_transformed)
        prediction.select("features", "prediction").show()

    # Example input data for prediction
    example_input = {
        'Glucose': 105, 'Cholesterol': 200, 'Hemoglobin': 13.5, 'Platelets': 250000, 'White Blood Cells': 8000,
        'Red Blood Cells': 4.5, 'Hematocrit': 40, 'Mean Corpuscular Volume': 80, 'Mean Corpuscular Hemoglobin': 27,
        'Mean Corpuscular Hemoglobin Concentration': 32, 'Insulin': 15, 'BMI': 22, 'Systolic Blood Pressure': 120,
        'Diastolic Blood Pressure': 80, 'Triglycerides': 150, 'HbA1c': 5.5, 'LDL Cholesterol': 100,
        'HDL Cholesterol': 50, 'ALT': 30, 'AST': 20, 'Heart Rate': 70, 'Creatinine': 1.0, 'Troponin': 0.01,
        'C-reactive Protein': 0.3
    }

    # Make a prediction for the example input
    make_prediction(example_input)

else:
    print("Train or Test Data is empty. Model training and evaluation cannot proceed.")

# Stop Spark session
spark.stop()


# COMMAND ----------


