from pyspark.ml.feature import VectorAssembler, StandardScaler


def assemble_features(df, input_cols, output_col="features"):
    assembler = VectorAssembler(inputCols=input_cols, outputCol="raw_features")
    df = assembler.transform(df)
    scaler = StandardScaler(inputCol="raw_features", outputCol=output_col)
    df = scaler.fit(df).transform(df)
    return df
