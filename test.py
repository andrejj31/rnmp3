def assemble_features(df, input_cols, output_col="features"):
    assembler = VectorAssembler(inputCols=input_cols, outputCol="raw_features")
    df = assembler.transform(df)
    # Scale features
    scaler = StandardScaler(inputCol="raw_features", outputCol=output_col)
    df = scaler.fit(df).transform(df)
    return df


# Apply transformations
feature_columns = [col for col in df.columns if col != "label"]
df_transformed = assemble_features(df, feature_columns)

# 4️⃣ Train/test split
train_df, test_df = df_transformed.randomSplit([0.8, 0.2], seed=42)

# 5️⃣ Define models with different hyperparameters
models = [
    ("LogisticRegression", LogisticRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.01)),
    ("RandomForest", RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20, maxDepth=5)),
    ("DecisionTree", DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=3))
]

# 6️⃣ Train models and evaluate
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

best_model = None
best_score = -1
best_name = ""

for name, model in models:
    # Pipeline
    pipeline = Pipeline(stages=[model])
    trained_model = pipeline.fit(train_df)

    predictions = trained_model.transform(test_df)
    f1 = evaluator.evaluate(predictions)
    print(f"Model {name} F1 score: {f1}")

    if f1 > best_score:
        best_score = f1
        best_model = trained_model
        best_name = name

# 7️⃣ Save best model
model_path = "/opt/models/best_model"
if not os.path.exists("/opt/models"):
    os.makedirs("/opt/models")

best_model.write().overwrite().save(model_path)
print(f"✅ Best model: {best_name} saved at {model_path} with F1={best_score}")

spark.stop()