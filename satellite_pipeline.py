"""
1. Reads all NPS wolf territory shapefiles from a base directory.
2. Unions them and adds a buffer to define a single area-of-interest (AOI).
3. For each year in a range:
   - Pulls Landsat Collection 2 Level-2 imagery (L5/L7/L8/L9).
   - Applies scale factors and cloud masking.
   - Computes spectral indices (NDVI, EVI, NDWI, NBR, BSI, NDSI).
   - Builds a growing-season (May-September) median composite over the AOI.
   - Exports a single GeoTIFF with all bands to Google Drive.
"""

import argparse
from pathlib import Path
from typing import List

import ee
import geopandas as gpd
from shapely.ops import unary_union


def load_all_territories(shapefile_root: str) -> gpd.GeoDataFrame:
    """load all .shp files under a root directory into a single GeoDataFrame."""
    root = Path(shapefile_root)
    shp_paths: List[Path] = [p for p in root.rglob("*.shp") if ".zip" not in str(p)]
    if not shp_paths:
        raise RuntimeError(f"No .shp files found under {shapefile_root}")

    records = []
    for shp in shp_paths:
        try:
            gdf = gpd.read_file(shp)
            if gdf.empty:
                continue
            if gdf.crs is None:
                gdf.set_crs("EPSG:26912", inplace=True)
            elif gdf.crs.to_epsg() != 26912:
                gdf = gdf.to_crs("EPSG:26912")
            for geom in gdf.geometry:
                if geom is not None:
                    records.append({"geometry": geom})
        except Exception as e:
            print(f"[WARN] Failed to read {shp}: {e}")

    if not records:
        raise RuntimeError("No valid geometries loaded from shapefiles.")

    return gpd.GeoDataFrame(records, crs="EPSG:26912")


def build_study_area(
    territories: gpd.GeoDataFrame, buffer_km: float = 15.0
) -> ee.Geometry:
    """union all territories, buffer, and convert to an Earth Engine geometry."""
    unified = unary_union(territories.geometry)
    buffered = unified.buffer(buffer_km * 1000.0)
    hull = buffered.convex_hull

    gdf = gpd.GeoDataFrame({"geometry": [hull]}, crs="EPSG:26912")
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf_wgs84.geometry[0].bounds

    print(f"Study area bounds (WGS84): ({minx}, {miny}, {maxx}, {maxy})")

    return ee.Geometry.Rectangle(
        [minx, miny, maxx, maxy], proj="EPSG:4326", geodesic=False
    )


LANDSAT_COLLECTIONS = {
    "L5": "LANDSAT/LT05/C02/T1_L2",
    "L7": "LANDSAT/LE07/C02/T1_L2",
    "L8": "LANDSAT/LC08/C02/T1_L2",
    "L9": "LANDSAT/LC09/C02/T1_L2",
}


def get_landsat_sensors_for_year(year: int) -> List[str]:
    """return Landsat sensors to use for a given year (no HLS)."""
    sensors: List[str] = []
    if year <= 2011:
        sensors.append("L5")
    if year >= 1999:
        sensors.append("L7")
    if year >= 2013:
        sensors.append("L8")
    if year >= 2022:
        sensors.append("L9")
    return sensors


def rename_landsat_bands(img: ee.Image, sensor: str) -> ee.Image:
    """map Landsat C2 L2 band names to common names."""
    if sensor in ("L5", "L7"):
        mapping = {
            "SR_B1": "BLUE",
            "SR_B2": "GREEN",
            "SR_B3": "RED",
            "SR_B4": "NIR",
            "SR_B5": "SWIR1",
            "SR_B7": "SWIR2",
            "QA_PIXEL": "QA_PIXEL",
        }
    else:  # L8 / L9
        mapping = {
            "SR_B2": "BLUE",
            "SR_B3": "GREEN",
            "SR_B4": "RED",
            "SR_B5": "NIR",
            "SR_B6": "SWIR1",
            "SR_B7": "SWIR2",
            "QA_PIXEL": "QA_PIXEL",
        }
    src = list(mapping.keys())
    dst = list(mapping.values())
    return img.select(src, dst)


def apply_scale_and_cloud_mask(img: ee.Image) -> ee.Image:
    """apply C2 L2 scale factors and QA-based cloud mask."""
    # Scale surface reflectance
    optical = img.select(["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"])
    optical = optical.multiply(0.0000275).add(-0.2).clamp(0, 1)

    # QA cloud/shadow/snow mask (bits 1,3,4,5)
    qa = img.select("QA_PIXEL")
    dilated_cloud_bit = 1 << 1
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    snow_bit = 1 << 5

    mask = (
        qa.bitwiseAnd(dilated_cloud_bit)
        .eq(0)
        .And(qa.bitwiseAnd(cloud_bit).eq(0))
        .And(qa.bitwiseAnd(shadow_bit).eq(0))
        .And(qa.bitwiseAnd(snow_bit).eq(0))
    )

    img = img.addBands(optical, overwrite=True)
    img = img.updateMask(mask)
    return img


def add_indices(img: ee.Image) -> ee.Image:
    """add NDVI, EVI, NDWI, NBR, BSI, NDSI bands."""
    nir = img.select("NIR")
    red = img.select("RED")
    green = img.select("GREEN")
    blue = img.select("BLUE")
    swir1 = img.select("SWIR1")
    swir2 = img.select("SWIR2")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": nir, "RED": red, "BLUE": blue},
    ).rename("EVI")

    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename("NBR")

    bsi = img.expression(
        "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
        {"SWIR1": swir1, "RED": red, "NIR": nir, "BLUE": blue},
    ).rename("BSI")

    ndsi = green.subtract(swir1).divide(green.add(swir1)).rename("NDSI")

    return img.addBands([ndvi, evi, ndwi, nbr, bsi, ndsi])


def get_landsat_collection(
    sensor: str,
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_threshold: float = 60.0,
) -> ee.ImageCollection:
    """landsat C2 L2 collection with renaming, scaling, cloud mask, indices."""
    coll_id = LANDSAT_COLLECTIONS[sensor]

    def process(img):
        img = rename_landsat_bands(img, sensor)
        img = apply_scale_and_cloud_mask(img)
        img = add_indices(img)
        return img

    return (
        ee.ImageCollection(coll_id)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_threshold))
        .map(process)
    )


def make_yearly_composite(year: int, aoi: ee.Geometry) -> ee.Image:
    """create a median growing-season composite for one year (Landsat only)."""
    start = f"{year}-05-01"
    end = f"{year}-09-30"

    sensors = get_landsat_sensors_for_year(year)
    if not sensors:
        raise RuntimeError(f"No Landsat sensors configured for year {year}")

    colls = [get_landsat_collection(s, aoi, start, end) for s in sensors]

    merged = colls[0]
    for c in colls[1:]:
        merged = merged.merge(c)

    bands = [
        "BLUE",
        "GREEN",
        "RED",
        "NIR",
        "SWIR1",
        "SWIR2",
        "NDVI",
        "EVI",
        "NDWI",
        "NBR",
        "BSI",
        "NDSI",
    ]

    composite = merged.select(bands).median()

    pixel_count = merged.select("NDVI").count().toFloat().rename("pixel_count")
    composite = composite.addBands(pixel_count)

    composite = composite.toFloat().clip(aoi)

    composite = composite.set(
        {
            "year": year,
            "start_date": start,
            "end_date": end,
            "sensors": ",".join(sensors),
            "composite": "median_growing_season",
        }
    )
    return composite


def export_composite_to_drive(
    img: ee.Image,
    year: int,
    aoi: ee.Geometry,
    drive_folder: str = "WolfCast_Simple",
    scale: int = 30,
):
    """start a Drive export for one year's composite."""
    desc = f"wolfcast_simple_{year}"
    prefix = f"wolfcast_simple_{year}"

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=desc,
        folder=drive_folder,
        fileNamePrefix=prefix,
        region=aoi,
        scale=scale,
        crs="EPSG:26912",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Started export for {year}: task_id={task.id}")
    return task


def main():
    parser = argparse.ArgumentParser(
        description="Simple WolfCast Landsat median composite pipeline (no HLS, no config file)."
    )
    parser.add_argument(
        "--shapefile-root",
        type=str,
        default="1995_2022-YELL-wolf-data",
        help="Root directory containing wolf territory shapefiles.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1995,
        help="First year to process (inclusive).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2022,
        help="Last year to process (inclusive).",
    )
    parser.add_argument(
        "--drive-folder",
        type=str,
        default="WolfCast_Simple",
        help="Google Drive folder to receive GeoTIFFs.",
    )
    parser.add_argument(
        "--buffer-km",
        type=float,
        default=15.0,
        help="Buffer in km around union of all territories.",
    )
    args = parser.parse_args()

    print("Initializing Google Earth Engine...")
    ee.Initialize(project="wolfcast")

    print(f"Loading shapefiles from {args.shapefile_root} ...")
    territories = load_all_territories(args.shapefile_root)
    print(f"Loaded {len(territories)} territory polygons.")

    print("Building unified study area...")
    aoi = build_study_area(territories, buffer_km=args.buffer_km)

    tasks = []
    for year in range(args.start_year, args.end_year + 1):
        print(f"\n=== Year {year} ===")
        try:
            img = make_yearly_composite(year, aoi)
            task = export_composite_to_drive(
                img, year, aoi, drive_folder=args.drive_folder
            )
            tasks.append((year, task.id))
        except Exception as e:
            print(f"[ERROR] Year {year} failed: {e}")

    print("\nSubmitted tasks:")
    for year, tid in tasks:
        print(f"  {year}: {tid}")
    print("Check progress at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
