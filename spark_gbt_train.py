from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import when, col

SAMPLES_PATH = "gs://wolfcast_training_samples/54m_3neg_1pos"
MODEL_OUTPUT_PATH = "gs://wolfcast_training_samples/models/habitat_model_54m_3neg_1pos"

TRAIN_YEARS = list(range(1995, 2016))
VAL_YEARS = list(range(2016, 2019))
TEST_YEARS = list(range(2019, 2023))


spark = SparkSession.builder \
    .appName("WolfHabitatTraining_v0") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

print(f"Reading samples from: {SAMPLES_PATH}")
df = spark.read.parquet(SAMPLES_PATH)
print(f"Total samples: {df.count()}")

train_df = df.filter(df.year.isin(TRAIN_YEARS))
val_df = df.filter(df.year.isin(VAL_YEARS))
test_df = df.filter(df.year.isin(TEST_YEARS))

print(f"Train: {train_df.count()}, Val: {val_df.count()}, Test: {test_df.count()}")

pos_count = train_df.filter(train_df.presence == 1).count()
neg_count = train_df.filter(train_df.presence == 0).count()
weight_ratio = neg_count / pos_count if pos_count > 0 else 1.0

train_df = train_df.withColumn(
    "weight",
    when(col("presence") == 1, weight_ratio).otherwise(1.0)
)

numeric_features = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2",
                    "NDVI", "EVI", "NDWI", "NBR", "BSI", "NDSI"]

indexer = StringIndexer(inputCol="nlcd_class", outputCol="nlcd_indexed", handleInvalid="keep")
encoder = OneHotEncoder(inputCols=["nlcd_indexed"], outputCols=["nlcd_encoded"], dropLast=True)

feature_cols = numeric_features + ["nlcd_encoded"]
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

gbt = GBTClassifier(
    labelCol="presence",
    featuresCol="features",
    weightCol="weight",
    maxDepth=10,
    maxIter=100,
    stepSize=0.1,
    seed=42
)

pipeline = Pipeline(stages=[indexer, encoder, assembler, gbt])

print("Training model...")
model = pipeline.fit(train_df)

val_predictions = model.transform(val_df)
evaluator = BinaryClassificationEvaluator(
    labelCol="presence",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
val_auc = evaluator.evaluate(val_predictions)
print(f"Validation AUC: {val_auc:.4f}")

test_predictions = model.transform(test_df)
test_auc = evaluator.evaluate(test_predictions)
print(f"Test AUC: {test_auc:.4f}")

model.write().overwrite().save(MODEL_OUTPUT_PATH)
print(f"Model saved to: {MODEL_OUTPUT_PATH}")

spark.stop()
