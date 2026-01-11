from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("OfflineTraining") \
    .master("local[*]") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .getOrCreate()

df = spark.read.csv("/Users/andrej/Desktop/rnmp/RNMP_homework1/data/offline.csv", header=True, inferSchema=True)

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION:")
df.groupBy("Diabetes_012").count().orderBy("Diabetes_012").show()
print("=" * 60 + "\n")

feature_columns = [col for col in df.columns if col != "Diabetes_012"]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

models = [
    ("LogisticRegression",
     LogisticRegression(featuresCol="features", labelCol="Diabetes_012", maxIter=20, regParam=0.01)),
    ("RandomForest", RandomForestClassifier(featuresCol="features", labelCol="Diabetes_012", numTrees=50, maxDepth=10)),
    ("DecisionTree", DecisionTreeClassifier(featuresCol="features", labelCol="Diabetes_012", maxDepth=10))
]

evaluator = MulticlassClassificationEvaluator(labelCol="Diabetes_012", predictionCol="prediction", metricName="f1")

best_model = None
best_score = -1
best_name = ""

for name, model in models:
    pipeline = Pipeline(stages=[assembler, scaler, model])
    trained_model = pipeline.fit(train_df)

    predictions = trained_model.transform(test_df)
    f1 = evaluator.evaluate(predictions)

    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)

    print(f"\n{'=' * 50}")
    print(f"Model: {name}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print(f"Prediction distribution:")
    predictions.groupBy("prediction").count().orderBy("prediction").show()

    if f1 > best_score:
        best_score = f1
        best_model = trained_model
        best_name = name

model_path = "../models/best_model"
best_model.write().overwrite().save(model_path)
print(f"\n Best model: {best_name} saved with F1={best_score:.4f}")

spark.stop()