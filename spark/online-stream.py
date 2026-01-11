from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import struct, col, from_json, to_json
from pyspark.sql.types import StructType, StructField, DoubleType

TOPIC_NAME = "health-data"
OUTPUT_TOPIC = "health-data-predicted"
KAFKA_BOOTSTRAP_SERVER = "localhost:9092"

spark = SparkSession.builder \
    .appName("OnlineStreaming") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

model_path = "/Users/andrej/Desktop/rnmp/RNMP_homework1/models/best_model"
model = PipelineModel.load(model_path)

raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVER) \
    .option("subscribe", TOPIC_NAME) \
    .option("startingOffsets", "latest") \
    .load()

schema = StructType([
    StructField("HighBP", DoubleType()),
    StructField("HighChol", DoubleType()),
    StructField("CholCheck", DoubleType()),
    StructField("BMI", DoubleType()),
    StructField("Smoker", DoubleType()),
    StructField("Stroke", DoubleType()),
    StructField("HeartDiseaseorAttack", DoubleType()),
    StructField("PhysActivity", DoubleType()),
    StructField("Fruits", DoubleType()),
    StructField("Veggies", DoubleType()),
    StructField("HvyAlcoholConsump", DoubleType()),
    StructField("AnyHealthcare", DoubleType()),
    StructField("NoDocbcCost", DoubleType()),
    StructField("GenHlth", DoubleType()),
    StructField("MentHlth", DoubleType()),
    StructField("PhysHlth", DoubleType()),
    StructField("DiffWalk", DoubleType()),
    StructField("Sex", DoubleType()),
    StructField("Age", DoubleType()),
    StructField("Education", DoubleType()),
    StructField("Income", DoubleType()),
])

stream_df = raw_stream.selectExpr("CAST(value AS STRING) as json_str")
parsed_df = stream_df.select(from_json(col("json_str"), schema).alias("data")).select("data.*")

predictions = model.transform(parsed_df)

feature_columns = [field.name for field in schema.fields]

output_df = predictions.withColumn(
    "value",
    to_json(
        struct(
            *[col(c) for c in feature_columns],
            col("prediction").alias("predicted_class"),
            col("probability").alias("prediction_probability")
        )
    )
).selectExpr("CAST(value AS STRING)")

kafka_query = output_df.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVER) \
    .option("topic", OUTPUT_TOPIC) \
    .option("checkpointLocation", "/tmp/kafka-checkpoint") \
    .outputMode("append") \
    .start()

spark.streams.awaitAnyTermination()