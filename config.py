"""
Shared path configuration for the whole pipeline.

All paths are relative to the repository root by default, so no script needs editing.
To keep the data elsewhere, set the environment variable UAV_DATA_DIR, e.g.
    export UAV_DATA_DIR=/path/to/my/data          (Linux / macOS)
    set UAV_DATA_DIR=D:\\uav_data                  (Windows cmd)

Expected layout (see data/README.md for download instructions):

    <DATA_DIR>/
      raw/                         <- inputs you download yourself
        boundary/Manchester_boundary.shp
        fsa/FHRS415en-GB.xml                               (FSA open data, Manchester)
        worldpop/Manchester_WorldPop_100m_2020.tif
        os_building_heights/download_1/{sd,sj,sk}/*.gdb   (OS Building Heights tiles)
        os_building_heights/download_2/...
        os_building_heights/download_3/...
        osm/gis_osm_roads_free_1.shp, gis_osm_railways_free_1.shp
        caa/EG_UAS_FR_DS_AREA1_FULL*.kmz                  (UK AIP restricted areas)
        icao/EG_OBS_DS_AREA1_FULL*.xls                    (ICAO Area 1 obstacles)
        wind/*.nc                                          (HadUK-Grid wind speed)
      processed/                   <- written by the scripts
"""

import glob
import os

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('UAV_DATA_DIR', os.path.join(ROOT, 'data'))
RAW      = os.path.join(DATA_DIR, 'raw')
PROC     = os.path.join(DATA_DIR, 'processed')

TARGET_CRS = 'EPSG:27700'   # British National Grid


def first_match(pattern):
    """Return the first file matching a glob pattern (for dated download names)."""
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else pattern


# ---- Raw inputs ---------------------------------------------------------
BOUNDARY    = os.path.join(RAW, 'boundary', 'Manchester_boundary.shp')
FSA_XML     = os.path.join(RAW, 'fsa', 'FHRS415en-GB.xml')   # FSA open data, Manchester (LA 415)
WORLDPOP    = os.path.join(RAW, 'worldpop', 'Manchester_WorldPop_100m_2020.tif')
BH_DIRS     = [os.path.join(RAW, 'os_building_heights', f'download_{i}') for i in (1, 2, 3)]
OSM_ROADS   = os.path.join(RAW, 'osm', 'gis_osm_roads_free_1.shp')
OSM_RAIL    = os.path.join(RAW, 'osm', 'gis_osm_railways_free_1.shp')
CAA_KMZ     = first_match(os.path.join(RAW, 'caa', 'EG_UAS_FR_DS_AREA1_FULL*.kmz'))
ICAO_XLS    = first_match(os.path.join(RAW, 'icao', 'EG_OBS_DS_AREA1_FULL*.xls'))
WIND_DIR    = os.path.join(RAW, 'wind')

# ---- Processed outputs (one folder per stage) ----------------------------
RESTAURANT_DIR   = os.path.join(PROC, 'restaurants')
CDI_DIR          = os.path.join(PROC, 'cdi')
BUILDINGS_DIR    = os.path.join(PROC, 'buildings')
VERTIPORT_DIR    = os.path.join(PROC, 'vertiports')
COST_DIR         = os.path.join(PROC, 'cost_maps')
ROUTES_DIR       = os.path.join(PROC, 'routes')
EVAL_DIR         = os.path.join(PROC, 'evaluation')

FSA_CSV          = os.path.join(RESTAURANT_DIR, 'manchester_fsa_complete.csv')
RATINGS_CSV      = os.path.join(RESTAURANT_DIR, 'manchester_restaurants_final.csv')
RESTAURANTS_CSV  = os.path.join(RESTAURANT_DIR, 'manchester_restaurants_recovered.csv')
CDI_TIF          = os.path.join(CDI_DIR, 'CDI_50m.tif')
CANDIDATES_SHP   = os.path.join(VERTIPORT_DIR, 'vertiport_candidates.shp')
CURVES_CSV       = os.path.join(VERTIPORT_DIR, 'coverage_curves.csv')
VERTIPORT_SHP    = os.path.join(VERTIPORT_DIR, 'vertiports_selected.shp')


def cost_dir(alt):
    """Folder holding the 5 m cost-map rasters for one altitude layer."""
    return os.path.join(COST_DIR, f'voxel_output_{alt}m')
