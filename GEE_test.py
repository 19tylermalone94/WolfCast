import ee

# Authenticate and initialize
ee.Authenticate(auth_mode="localhost")
ee.Initialize(project="wolfcast")

# Define area of interest (AOI)
aoi = ee.Geometry.Point([-110.5519444444, 44.6922222222]).buffer(50000)  # 50 km radius around a point in yellowstone

# Can use a polygon instead of the above which is a circle
# aoi = ee.Geometry.Polygon([
#     [-110.5519444444, 44.6922222222],
#     [-110.5519444444, 44.6922222222],

# Load a surface reflectance image collection
# Sentinel-2 only goes back to 2015; to cover 1995–2024, use Landsat.
collection = (
    ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")  # Landsat 5 (1984–2013)
    .merge(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"))  # Landsat 7 (1999–present)
    .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))  # Landsat 8 (2013–present)
    .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))  # Landsat 9 (2021–present)
    .filterBounds(aoi)
    .filterDate("1995-01-01", "2024-12-31")
    .filterMetadata("SUN_ELEVATION", "greater_than", 10)  # roughly daytime
    .sort("CLOUD_COVER")  # least cloudy first
)

# Count matching images
count = collection.size().getInfo()
print(f"Number of matching images found: {count}")

# Take first 3 images
subset = collection.limit(3)

# Loop through and export 3 example images
for i, img in enumerate(subset.toList(3).getInfo()):
    img_id = img["id"]
    image = ee.Image(img_id)
    rgb = image.select(["SR_B4", "SR_B3", "SR_B2"])  # RGB for Landsat
    export_task = ee.batch.Export.image.toDrive(
        image=rgb.clip(aoi),
        description=f"wolfcast_image_{i+1}",
        folder="EarthEngine",
        fileNamePrefix=f"wolfcast_image_{i+1}",
        scale=30,
        region=aoi.getInfo()["coordinates"],
    )
    export_task.start()
    print(f"Export started for image {i+1}: {img_id}")

print("Images exported to EarthEngine folder in your Google Drive")
