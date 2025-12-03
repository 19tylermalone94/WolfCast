# WolfCast

## Google Earth Engine

1. Make a Python virtual environment. I used python3.13. To use school machine run `module load python/bundle-3.13` or just install it locally. Other python versions will probably work.

```
python -m venv venv
```
2. Source venv
```
source venv/bin/activate
```

3. Install GEE module
```
pip install --upgrade earthengine-api
```

4. Authenticate with your Google account, so it can upload images to your Google Drive. Run this in the command line, and it should open a browser to get your consent:
```
earthengine authenticate
```

5. You should be able to run the test script and find the images saved to the EarthEngine folder automatically created in your Google Drive:
```
python GEE_test.py
```

6. Check the EarthEngine folder in your Google Drive

## Running a Job
```
gcloud dataproc jobs submit pyspark \
  gs://wolfcast_training_samples/xgboost_trains.py \
  --cluster=cluster-wolf \
  --region=us-east1 \
  -- --config=gs://wolfcast_training_samples/configs/81m_1neg_1pos_v2.yaml
```
