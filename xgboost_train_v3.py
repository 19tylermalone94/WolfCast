from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql.functions import when, col
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from xgboost.spark import SparkXGBClassifier

SAMPLES_PATH = "gs://wolfcast_training_samples/training_samples_halfpos"
MODEL_OUTPUT_PATH = "gs://wolfcast_training_samples/models/habitat_xgb_halfpos"

TRAIN_YEARS = list(range(1995, 2016))
VAL_YEARS = list(range(2016, 2019))
TEST_YEARS = list(range(2019, 2023))

spark = (
    SparkSession.builder
    .appName("WolfHabitatTraining_XGB_v2")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.dynamicAllocation.enabled", "false")
    .config("spark.executor.instances", "7")
    .config("spark.executor.cores", "1")
    .config("spark.task.cpus", "1")
    .getOrCreate()
)

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

numeric_features = [
    "BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2",
    "NDVI", "EVI", "NDWI", "NBR", "BSI", "NDSI"
]

indexer = StringIndexer(
    inputCol="nlcd_class",
    outputCol="nlcd_indexed",
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCols=["nlcd_indexed"],
    outputCols=["nlcd_encoded"],
    dropLast=True
)

feature_cols = numeric_features + ["nlcd_encoded"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

xgb = SparkXGBClassifier(
    label_col="presence",
    features_col="features",
    weight_col="weight",

    max_depth=10,
    eta=0.1,
    num_round=300,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    eval_metric="auc",

    num_workers=4,
    tree_method="hist",
    missing=0.0,
    seed=42
)

pipeline = Pipeline(stages=[indexer, encoder, assembler, xgb])

print("Training XGBoost model...")
model = pipeline.fit(train_df)

evaluator = BinaryClassificationEvaluator(
    labelCol="presence",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

val_pred = model.transform(val_df)
val_auc = evaluator.evaluate(val_pred)
print(f"Validation AUC: {val_auc:.4f}")

test_pred = model.transform(test_df)
test_auc = evaluator.evaluate(test_pred)
print(f"Test AUC: {test_auc:.4f}")

model.write().overwrite().save(MODEL_OUTPUT_PATH)
print(f"Model saved to: {MODEL_OUTPUT_PATH}")

spark.stop()
