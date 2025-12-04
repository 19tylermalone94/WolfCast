import ee

ee.Authenticate(auth_mode="localhost")
ee.Initialize(project="wolfcast")

# --- helper: make a true square in meters around a lon/lat ---
def make_square(lon, lat, width_m=20000):
  # buffer half-width (meters) and take the bounding box -> square in meters
  half = width_m / 2
  return ee.Geometry.Point([lon, lat]).buffer(half).bounds()

# Base Yellowstone point
base_lon, base_lat = -110.5519444444, 44.6922222222

# Offsets in meters for nearby AOIs (about ~22 km lat shift ≈ 0.2° at this latitude)
# Using meters avoids degree distortion; you can tweak these.
def offset_point(lon, lat, dlon_deg=0, dlat_deg=0):
  return (lon + dlon_deg, lat + dlat_deg)

# Keep “few different points” but build true-square AOIs at each
c_lon, c_lat = base_lon, base_lat
n_lon, n_lat = base_lon, base_lat + 0.2
s_lon, s_lat = base_lon, base_lat - 0.2
e_lon, e_lat = base_lon + 0.2, base_lat
w_lon, w_lat = base_lon - 0.2, base_lat

aoi_list = [
  make_square(c_lon, c_lat, 20000),  # ~20 km x 20 km
  make_square(n_lon, n_lat, 20000),
  make_square(s_lon, s_lat, 20000),
  make_square(e_lon, e_lat, 20000),
  make_square(w_lon, w_lat, 20000),
]

# HLS v2.0 collections
hlsl30 = ee.ImageCollection("NASA/HLS/HLSL30/v002")
hlss30 = ee.ImageCollection("NASA/HLS/HLSS30/v002")

for idx, aoi in enumerate(aoi_list):
  collection = (
    hlsl30.merge(hlss30)
    .filterBounds(aoi)
    .filterDate("2013-01-01", "2024-12-31")
    .filter(ee.Filter.lt("CLOUD_COVERAGE", 60))
    .sort("CLOUD_COVERAGE")
  )

  count = collection.size().getInfo()
  print(f"AOI {idx+1}: {count} HLS images found")

  subset = collection.limit(3)

  # Export 3 least-cloudy images as square RGB with square pixel grid
  # Bands B4,B3,B2 are valid for both HLSL30/HLSS30 per GEE catalog.
  for i, img in enumerate(subset.toList(3).getInfo()):
    img_id = img["id"]
    image = ee.Image(img_id).select(["B4", "B3", "B2"])

    export_task = ee.batch.Export.image.toDrive(
      image=image.clip(aoi),
      description=f"wolfcast_hls_square_{idx+1}_{i+1}",
      folder="EarthEngine",
      fileNamePrefix=f"wolfcast_hls_square_{idx+1}_{i+1}",
      scale=30,                       # 30 m HLS pixels
      region=aoi.getInfo()["coordinates"],  # square region
      dimensions="1024x1024",         # force a square raster output
      maxPixels=1_000_000_000
    )
    export_task.start()
    print(f"Export started for AOI {idx+1}, image {i+1}: {img_id}")

print("All images exported to your EarthEngine folder in Drive.")
