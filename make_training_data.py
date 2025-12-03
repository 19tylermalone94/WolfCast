"""
Build training table from GCS TIFFs using Spark.

Samples pixels from yearly composites + presence masks + NLCD,
writes a Parquet table with features and labels for Spark training.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

# Configuration
GCS_BUCKET = "wolfcast_training_samples"
COMPOSITE_PREFIX = "data"
LABEL_PREFIX = "labels"
NLCD_GCS_PATH = f"gs://{GCS_BUCKET}/nlcd/wolfcast_nlcd_2019.tif"
OUTPUT_PATH = f"gs://{GCS_BUCKET}/training_samples_halfpos"

START_YEAR = 1995
END_YEAR = 2022

NEGATIVE_RATIO = 1.0
MAX_POSITIVES_PER_YEAR = 2000000

BAND_NAMES = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
INDEX_NAMES = ["NDVI", "EVI", "NDWI", "NBR", "BSI", "NDSI"]

# Schema for Spark DataFrame
schema = StructType(
    [
        StructField("year", IntegerType(), False),
        StructField("presence", IntegerType(), False),
        StructField("nlcd_class", IntegerType(), False),
    ]
    + [StructField(name, DoubleType(), False) for name in BAND_NAMES + INDEX_NAMES]
)


def process_year(year):
    """Process a year on an executor - returns list of row dicts."""
    import tempfile
    from pathlib import Path
    import numpy as np
    import rasterio
    from google.cloud import storage
    
    # Define all constants inside function to avoid serialization issues
    GCS_BUCKET = "wolfcast_training_samples"
    COMPOSITE_PREFIX = "data"
    LABEL_PREFIX = "labels"
    MAX_POSITIVES_PER_YEAR = 2000000
    NEGATIVE_RATIO = 1.0
    BAND_NAMES = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
    INDEX_NAMES = ["NDVI", "EVI", "NDWI", "NBR", "BSI", "NDSI"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        comp_local = tmp_path / f"comp_{year}.tif"
        label_local = tmp_path / f"label_{year}.tif"
        nlcd_local = tmp_path / "nlcd.tif"
        
        try:
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            bucket.blob(f"{COMPOSITE_PREFIX}/wolfcast_{year}.tif").download_to_filename(str(comp_local))
            bucket.blob(f"{LABEL_PREFIX}/wolf_presence_{year}.tif").download_to_filename(str(label_local))
            bucket.blob("nlcd/wolfcast_nlcd_2019.tif").download_to_filename(str(nlcd_local))
        except Exception as e:
            return []
        
        try:
            with rasterio.open(comp_local) as comp_src, \
                 rasterio.open(label_local) as label_src, \
                 rasterio.open(nlcd_local) as nlcd_src:
                
                comp_data = comp_src.read()
                label_data = label_src.read(1)
                nlcd_data = nlcd_src.read(1)
                
                h, w = label_data.shape
                comp_data = comp_data[:, :h, :w]
                nlcd_data = nlcd_data[:h, :w]
                
                n_pixels = h * w
                n_bands = comp_data.shape[0]
                comp_flat = comp_data.reshape(n_bands, n_pixels).T
                label_flat = label_data.flatten()
                nlcd_flat = nlcd_data.flatten()
                
                valid_mask = ~np.isnan(comp_flat).any(axis=1) & (nlcd_flat > 0)
                
                if not valid_mask.any():
                    return []
                
                comp_valid = comp_flat[valid_mask]
                label_valid = label_flat[valid_mask]
                nlcd_valid = nlcd_flat[valid_mask]
                
                pos_mask = label_valid == 1
                neg_mask = label_valid == 0
                
                pos_indices = np.where(pos_mask)[0]
                neg_indices = np.where(neg_mask)[0]
                
                if len(pos_indices) > MAX_POSITIVES_PER_YEAR:
                    pos_selected = np.random.choice(pos_indices, MAX_POSITIVES_PER_YEAR, replace=False)
                else:
                    pos_selected = pos_indices
                
                n_neg_target = int(len(pos_selected) * NEGATIVE_RATIO)
                if len(neg_indices) > n_neg_target:
                    neg_selected = np.random.choice(neg_indices, n_neg_target, replace=False)
                else:
                    neg_selected = neg_indices
                
                all_selected = np.concatenate([pos_selected, neg_selected])
                np.random.shuffle(all_selected)
                
                rows = []
                for idx in all_selected:
                    row = {
                        "year": year,
                        "presence": int(label_valid[idx]),
                        "nlcd_class": int(nlcd_valid[idx]),
                    }
                    for i, name in enumerate(BAND_NAMES):
                        row[name] = float(comp_valid[idx, i])
                    for i, name in enumerate(INDEX_NAMES):
                        row[name] = float(comp_valid[idx, i + len(BAND_NAMES)])
                    rows.append(row)
                
                return rows
        except Exception as e:
            return []


def main():
    spark = (
        SparkSession.builder
        .appName("WolfHabitatTraining")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.dynamicAllocation.enabled", "false")
        .config("spark.executor.instances", "7")
        .config("spark.executor.cores", "1")
        .config("spark.task.cpus", "1")
        .getOrCreate()
    )
    
    sc = spark.sparkContext
    years = list(range(START_YEAR, END_YEAR + 1))
    
    # Distribute years across executors
    years_rdd = sc.parallelize(years, numSlices=len(years))
    
    # Process each year on executors
    results_rdd = years_rdd.flatMap(process_year)
    
    # Collect and group by year
    all_results = results_rdd.collect()
    
    from collections import defaultdict
    by_year = defaultdict(list)
    for row in all_results:
        by_year[row["year"]].append(row)
    
    total_count = 0
    total_pos = 0
    total_neg = 0
    
    # Write each year
    for year, rows in sorted(by_year.items()):
        if not rows:
            continue
        
        df = spark.createDataFrame(rows, schema=schema)
        df = df.withColumn("random", rand()).orderBy("random").drop("random")
        
        year_path = f"{OUTPUT_PATH}/samples_{year}.parquet"
        df.write.mode("overwrite").parquet(year_path)
        
        n_pos = sum(1 for r in rows if r["presence"] == 1)
        n_neg = len(rows) - n_pos
        total_count += len(rows)
        total_pos += n_pos
        total_neg += n_neg
        
        print(f"[saved] {year}: {len(rows)} samples ({n_pos} pos, {n_neg} neg)")
    
    print(f"\n[total] {total_count} samples")
    print(f"  Positives: {total_pos}")
    print(f"  Negatives: {total_neg}")
    print(f"\n[saved] {OUTPUT_PATH}/*.parquet")
    print(f"Spark can read with: spark.read.parquet('{OUTPUT_PATH}')")
    
    spark.stop()


if __name__ == "__main__":
    main()
