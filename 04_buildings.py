"""
04_buildings.py
Merge three OS Building Height Attribute datasets covering Manchester,
including southern tiles (Altrincham / Airport) acquired separately.

Outputs:
  data/processed/buildings/os_buildings_manchester.shp
  data/processed/buildings/buildings_obstacle_{25,50,80,120}m.shp
"""

import os
import pandas as pd
import geopandas as gpd
import fiona
from shapely.ops import unary_union

import config

TARGET_CRS = config.TARGET_CRS
BOUNDARY   = config.BOUNDARY
OUT_DIR    = config.BUILDINGS_DIR
os.makedirs(OUT_DIR, exist_ok=True)

BH_DIRS = config.BH_DIRS   # OS Building Heights downloads (tiles de-duplicated by name)

OBSTACLE_THRESHOLDS = [25, 50, 80, 120]

def log(msg):    print(f"\n{'='*55}\n{msg}\n{'='*55}")
def sublog(msg): print(f"  -> {msg}")


log("Step 1: Load boundary")
boundary  = gpd.read_file(BOUNDARY).to_crs(TARGET_CRS)
bnd_union = unary_union(boundary.geometry)
sublog(f"Area: {bnd_union.area/1e6:.2f} km2")


log("Step 2: Load GDB tiles (deduplicated by tile name)")
all_gdfs   = {}
tile_count = 0

for bh_base in BH_DIRS:
    if not os.path.exists(bh_base):
        sublog(f"Directory not found, skipping: {bh_base}")
        continue
    for folder in ['sd', 'sj', 'sk']:
        fp = os.path.join(bh_base, folder)
        if not os.path.exists(fp):
            continue
        for item in sorted(os.listdir(fp)):
            if not item.endswith('.gdb'):
                continue
            tile_name = item.replace('.gdb', '')
            if tile_name in all_gdfs:
                continue  # skip duplicate tiles present in multiple downloads
            gdb_path = os.path.join(fp, item)
            try:
                layers   = fiona.listlayers(gdb_path)
                gdf_tile = gpd.read_file(gdb_path, layer=layers[0])
                all_gdfs[tile_name] = gdf_tile
                tile_count += 1
                sublog(f"  {tile_name}: {len(gdf_tile)} buildings")
            except Exception as e:
                sublog(f"  Skipped {tile_name}: {e}")

sublog(f"Unique tiles loaded: {tile_count}")


log("Step 3: Merge and clip to Manchester boundary")
bh_all = gpd.GeoDataFrame(
    pd.concat(all_gdfs.values(), ignore_index=True),
    crs=TARGET_CRS)
sublog(f"Combined total: {len(bh_all):,}")

bh_mcr = gpd.clip(bh_all, boundary)
sublog(f"Within boundary: {len(bh_mcr):,}")

keep_cols = ['os_topo_toid', 'relhmax', 'relh2',
             'abshmax', 'abshmin', 'bha_conf', 'geometry']
keep_cols = [c for c in keep_cols if c in bh_mcr.columns]
bh_mcr    = bh_mcr[keep_cols].copy()


log("Step 4: Height statistics")
sublog(f"relhmax mean: {bh_mcr['relhmax'].mean():.1f} m")
sublog(f"relhmax max:  {bh_mcr['relhmax'].max():.1f} m")
for thresh in OBSTACLE_THRESHOLDS:
    n = (bh_mcr['relhmax'] > thresh).sum()
    sublog(f"  > {thresh} m: {n:,}")


log("Step 5: Save")
bh_mcr.to_file(os.path.join(OUT_DIR, 'os_buildings_manchester.shp'))
sublog(f"All buildings saved ({len(bh_mcr):,})")

for thresh in OBSTACLE_THRESHOLDS:
    subset = bh_mcr[bh_mcr['relhmax'] > thresh].copy()
    fname  = f'buildings_obstacle_{thresh}m.shp'
    subset.to_file(os.path.join(OUT_DIR, fname))
    sublog(f"Obstacle > {thresh} m: {fname} ({len(subset):,})")

log("Done")
