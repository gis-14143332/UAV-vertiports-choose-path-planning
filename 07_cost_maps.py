"""
07_cost_maps.py
Build altitude-specific soft constraint cost maps (25 / 50 / 80 / 120 m AGL).

Soft constraint layers (unified AHP weights; Saaty, 1980):
  Wind turbulence     w = 0.35  HadUK-Grid 7-year mean; primary UAV risk factor
                                (Bayesian network analysis, Hao et al., 2026)
  Proximity buffer    w = 0.30  Gradient decay: 1 - min(d / 50 m, 1.0)
                                (Ochoa & Atkins, JAIS)
  GNSS occlusion      w = 0.20  Distance to buildings > 10 m relhmax; 200 m decay
                                (Delamer et al., 2021)
  Light rail EMI      w = 0.15  50 m gradient buffer around Metrolink
                                (Kubacki et al., 2025)

Proximity buffer uses only altitude-specific obstacle buildings (relhmax > altitude).
GNSS uses buildings > 10 m relhmax to avoid overestimating occlusion from low-rise stock.
Hard constraints are handled as vector polygons in Phase 5 (no rasterisation).
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from shapely.ops import unary_union
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
import config

BOUNDARY  = config.BOUNDARY
BLD_BASE  = config.BUILDINGS_DIR
ICAO_XLS  = config.ICAO_XLS
OSM_RAIL  = config.OSM_RAIL
WIND_DIR  = config.WIND_DIR
VERTIPORT = config.VERTIPORT_SHP

# ============================================================
# Parameters
# ============================================================
TARGET_CRS  = config.TARGET_CRS
RESOLUTION  = 5           # grid cell size (m)
ALTITUDES   = [25, 50, 80, 120]
PROX_THRESH = 50.0        # proximity buffer decay distance (m)
RAIL_BUFFER = 50.0        # light rail EMI buffer (m)
GNSS_DECAY  = 200.0       # GNSS occlusion decay distance (m)
GNSS_MIN_H  = 10.0        # minimum building height for GNSS computation (m)
W_WIND      = 0.35
W_PROX      = 0.30
W_GNSS      = 0.20
W_RAIL      = 0.15

# ============================================================
# Utility
# ============================================================
def log(msg):    print(f"\n{'='*60}\n{msg}\n{'='*60}")
def sublog(msg): print(f"  -> {msg}")

def minmax_norm(arr, mask=None):
    valid = arr[mask & ~np.isnan(arr)] if mask is not None else arr[~np.isnan(arr)]
    if len(valid) == 0 or valid.max() == valid.min():
        return np.zeros_like(arr, dtype='float32')
    mn, mx = float(valid.min()), float(valid.max())
    result = arr.astype('float32')
    result = (result - mn) / (mx - mn)
    np.clip(result, 0, 1, out=result)
    return result

def save_raster(array, transform, crs, path, nodata=-9999):
    out = np.where(np.isnan(array), nodata, array).astype('float32')
    with rasterio.open(path, 'w', driver='GTiff',
                       height=array.shape[0], width=array.shape[1],
                       count=1, dtype='float32', crs=crs,
                       transform=transform, nodata=nodata, compress='lzw') as dst:
        dst.write(out, 1)

def kdtree_dist(pts_query, pts_ref, nrows, ncols, valid_idx):
    """Distance from each valid grid cell to its nearest reference point."""
    tree = cKDTree(pts_ref)
    dists, _ = tree.query(pts_query, k=1, workers=-1)
    result = np.full(nrows*ncols, 9999.0, dtype='float32')
    result[valid_idx] = dists.astype('float32')
    return result.reshape(nrows, ncols)


# ============================================================
# Step 1: Boundary and 5 m grid
# ============================================================
log("Step 1: Boundary and 5 m grid")
boundary  = gpd.read_file(BOUNDARY).to_crs(TARGET_CRS)
bnd_union = unary_union(boundary.geometry)
bounds    = boundary.total_bounds
minx, miny, maxx, maxy = bounds
minx = np.floor(minx/RESOLUTION)*RESOLUTION
miny = np.floor(miny/RESOLUTION)*RESOLUTION
maxx = np.ceil(maxx/RESOLUTION)*RESOLUTION
maxy = np.ceil(maxy/RESOLUTION)*RESOLUTION
ncols = int((maxx - minx) / RESOLUTION)
nrows = int((maxy - miny) / RESOLUTION)
transform_5m = from_bounds(minx, miny, maxx, maxy, ncols, nrows)

bnd_mask = geometry_mask(
    [mapping(g) for g in boundary.geometry],
    transform=transform_5m, invert=True, out_shape=(nrows, ncols))
sublog(f"Grid: {nrows}x{ncols}, valid cells: {bnd_mask.sum():,}")

# Pre-compute valid cell centre coordinates (shared across all altitude layers)
cols_idx  = np.arange(ncols)
rows_idx  = np.arange(nrows)
cc, rr    = np.meshgrid(cols_idx, rows_idx)
grid_x    = (minx + cc*RESOLUTION + RESOLUTION/2).astype('float32')
grid_y    = (maxy - rr*RESOLUTION - RESOLUTION/2).astype('float32')
valid_idx = np.where(bnd_mask.ravel())[0]
valid_pts = np.column_stack([grid_x.ravel()[valid_idx],
                              grid_y.ravel()[valid_idx]])
del cc, rr
sublog(f"Valid cells: {len(valid_idx):,}")


# ============================================================
# Step 2: Wind turbulence risk (shared across all altitude layers)
# HadUK-Grid 7-year mean sfcWind, bilinearly resampled to 5 m
# ============================================================
log("Step 2: Wind turbulence risk (HadUK-Grid 2019-2025)")
wind_risk   = None
wind_loaded = False

nc_files = sorted([os.path.join(WIND_DIR, f)
                   for f in os.listdir(WIND_DIR)
                   if f.lower().endswith('.nc')])
sublog(f"NetCDF files: {len(nc_files)}")

try:
    import netCDF4 as nc_lib
    wind_arrays    = []
    wind_transform = None
    for wf in nc_files:
        ds = nc_lib.Dataset(wf)
        wind_var = next((v for v in ['sfcWind', 'wind_speed', 'windspeed', 'ws']
                         if v in ds.variables), None)
        if wind_var:
            data = ds.variables[wind_var][:]
            if hasattr(data, 'filled'): data = data.filled(np.nan)
            arr  = np.nanmean(data, axis=0) if data.ndim == 3 else np.array(data)
            wind_arrays.append(arr)
            if wind_transform is None:
                x_var = next((v for v in ['projection_x_coordinate', 'easting', 'x']
                              if v in ds.variables), None)
                y_var = next((v for v in ['projection_y_coordinate', 'northing', 'y']
                              if v in ds.variables), None)
                if x_var and y_var:
                    x_arr = np.array(ds.variables[x_var][:])
                    y_arr = np.array(ds.variables[y_var][:])
                    dx    = float(x_arr[1] - x_arr[0]) if len(x_arr) > 1 else 1000
                    dy    = float(y_arr[1] - y_arr[0]) if len(y_arr) > 1 else 1000
                    y_north = float(y_arr[-1]) + abs(dy)/2 if dy > 0 else float(y_arr[0]) + abs(dy)/2
                    from rasterio.transform import from_origin
                    wind_transform = from_origin(float(x_arr[0]) - abs(dx)/2,
                                                 y_north, abs(dx), abs(dy))
        ds.close()

    if wind_arrays and wind_transform:
        wind_mean = np.nanmean(np.stack(wind_arrays, axis=0), axis=0)
        wind_5m   = np.zeros((nrows, ncols), dtype='float32')
        with rasterio.MemoryFile() as memfile:
            with memfile.open(driver='GTiff',
                              height=wind_mean.shape[0], width=wind_mean.shape[1],
                              count=1, dtype='float32', crs=TARGET_CRS,
                              transform=wind_transform) as tmp:
                tmp.write(wind_mean.astype('float32'), 1)
            with memfile.open() as tmp:
                reproject(source=rasterio.band(tmp, 1),
                          destination=wind_5m,
                          src_transform=wind_transform, src_crs=TARGET_CRS,
                          dst_transform=transform_5m, dst_crs=TARGET_CRS,
                          resampling=Resampling.bilinear)
        fill_val = float(np.nanmean(wind_5m[wind_5m > 0])) if np.any(wind_5m > 0) else 4.0
        wind_5m  = np.where(wind_5m <= 0, fill_val, wind_5m)
        wind_5m  = np.where(bnd_mask, wind_5m, np.nan).astype('float32')
        wind_norm = minmax_norm(wind_5m, bnd_mask)
        wind_risk  = np.where(bnd_mask, wind_norm, np.nan).astype('float32')
        wind_loaded = True
        sublog(f"HadUK mean wind speed: {np.nanmean(wind_5m):.2f} m/s")
        sublog(f"Wind risk: mean={np.nanmean(wind_risk):.4f}")
except Exception as e:
    sublog(f"HadUK load failed: {e}")

if not wind_loaded:
    sublog("Using uniform wind proxy (HadUK unavailable)")
    wind_risk = np.where(bnd_mask, 0.5, np.nan).astype('float32')


# ============================================================
# Step 3: GNSS occlusion risk (shared across all altitude layers)
# Only buildings > GNSS_MIN_H (10 m) are included to avoid overestimating
# occlusion from low-rise residential stock (Delamer et al., 2021)
# ============================================================
log("Step 3: GNSS occlusion risk")
bld_all  = gpd.read_file(os.path.join(BLD_BASE, 'os_buildings_manchester.shp'))
bld_gnss = bld_all[bld_all['relhmax'] > GNSS_MIN_H].copy()
bld_cx   = bld_gnss.geometry.centroid.x.values.astype('float32')
bld_cy   = bld_gnss.geometry.centroid.y.values.astype('float32')
sublog(f"GNSS: {len(bld_gnss)} buildings > {GNSS_MIN_H} m "
       f"(of {len(bld_all)} total)")

dist_to_gnss_bld = kdtree_dist(
    valid_pts, np.column_stack([bld_cx, bld_cy]), nrows, ncols, valid_idx)
gnss_raw  = np.clip(1 - dist_to_gnss_bld / GNSS_DECAY, 0, 1).astype('float32')
gnss_risk = minmax_norm(gnss_raw, bnd_mask)
gnss_risk = np.where(bnd_mask, gnss_risk, np.nan).astype('float32')
sublog(f"GNSS risk: mean={np.nanmean(gnss_risk):.4f}")
del bld_all, bld_gnss, bld_cx, bld_cy, gnss_raw


# ============================================================
# Step 4: Light rail EMI buffer (shared across all altitude layers)
# 50 m gradient buffer around Metrolink tram / light rail network
# ============================================================
log("Step 4: Light rail EMI buffer (Metrolink)")
railways  = gpd.read_file(OSM_RAIL).to_crs(TARGET_CRS)
metrolink = railways[railways['fclass'].isin(['light_rail', 'tram'])].copy()
metrolink = gpd.clip(metrolink, boundary)
sublog(f"Metrolink segments: {len(metrolink)}")

if len(metrolink) > 0:
    rail_pts = []
    for geom in metrolink.geometry:
        if geom is not None:
            try:   rail_pts.extend(list(geom.coords))
            except: pass
    rail_pts  = np.array(rail_pts, dtype='float32')
    dist_rail = kdtree_dist(valid_pts, rail_pts, nrows, ncols, valid_idx)
    del rail_pts
    rail_risk = np.where(bnd_mask,
                         np.clip(1 - dist_rail/RAIL_BUFFER, 0, 1),
                         np.nan).astype('float32')
    sublog(f"Rail risk: mean={np.nanmean(rail_risk):.4f}")
else:
    rail_risk = np.where(bnd_mask, 0.0, np.nan).astype('float32')


# ============================================================
# Step 5: ICAO obstacle coordinates (shared proxy for proximity)
# ============================================================
log("Step 5: Load ICAO obstacles")
icao_pts = None
try:
    import re

    def parse_icao_coord(coord_str):
        coord_str = str(coord_str).replace('\r\n', ' ').strip()
        parts = coord_str.split()
        if len(parts) < 2: return None, None
        def dms(s):
            hemi = s[-1]; nums = s[:-1]
            int_p, s_dec = (nums.split('.') + ['0'])[:2]
            s_dec = float('0.' + s_dec) if s_dec else 0.0
            L = len(int_p)
            if L == 5:   d, m, sc = int(int_p[0]),   int(int_p[1:3]), int(int_p[3:5]) + s_dec
            elif L == 6: d, m, sc = int(int_p[0:2]), int(int_p[2:4]), int(int_p[4:6]) + s_dec
            elif L == 7: d, m, sc = int(int_p[0:3]), int(int_p[3:5]), int(int_p[5:7]) + s_dec
            else:        d, m, sc = int(int_p[0:2]), int(int_p[2:4]), float(int_p[4:])  + s_dec
            dd = d + m/60 + sc/3600
            if hemi in ('S', 'W'): dd = -dd
            return dd
        try:   return dms(parts[0]), dms(parts[1])
        except: return None, None

    icao_df  = pd.read_excel(ICAO_XLS, sheet_name='All')
    coords_p = icao_df['Obstacle Position'].apply(
        lambda x: pd.Series(parse_icao_coord(x), index=['lat', 'lon']))
    icao_df  = pd.concat([icao_df, coords_p], axis=1).dropna(subset=['lat', 'lon'])

    def parse_height(h):
        if pd.isna(h): return 0
        m  = re.search(r'([\d.]+)\s*FT', str(h), re.IGNORECASE)
        m2 = re.search(r'([\d.]+)\s*M',  str(h), re.IGNORECASE)
        if m:  return float(m.group(1)) * 0.3048
        if m2: return float(m2.group(1))
        return 0

    icao_df['height_m'] = icao_df['Height'].apply(parse_height)
    icao_df  = icao_df[icao_df['height_m'] > 30]
    icao_gdf = gpd.GeoDataFrame(
        icao_df,
        geometry=gpd.points_from_xy(icao_df['lon'], icao_df['lat']),
        crs='EPSG:4326').to_crs(TARGET_CRS)
    icao_gdf = gpd.clip(icao_gdf, boundary)
    if len(icao_gdf) > 0:
        icao_pts = np.column_stack([
            icao_gdf.geometry.x.values.astype('float32'),
            icao_gdf.geometry.y.values.astype('float32')])
        sublog(f"ICAO obstacles: {len(icao_gdf)}")
except Exception as e:
    sublog(f"ICAO load skipped: {e}")


# ============================================================
# Step 6: Per-altitude cost maps
# ============================================================
log("Step 6: Per-altitude cost maps")
sublog(f"Weights: wind={W_WIND}, proximity={W_PROX}, GNSS={W_GNSS}, rail={W_RAIL}")

try:
    vp_gdf = gpd.read_file(VERTIPORT).to_crs(TARGET_CRS)
except:
    vp_gdf = None

for alt in ALTITUDES:
    sublog(f"\n-- {alt} m AGL --")
    out_dir = config.cost_dir(alt)
    os.makedirs(out_dir, exist_ok=True)

    # Obstacle buildings for this altitude layer
    bld_obs = gpd.read_file(os.path.join(BLD_BASE, f'buildings_obstacle_{alt}m.shp'))
    sublog(f"  Obstacle buildings (relhmax > {alt} m): {len(bld_obs)}")

    obs_cx  = bld_obs.geometry.centroid.x.values.astype('float32')
    obs_cy  = bld_obs.geometry.centroid.y.values.astype('float32')
    obs_pts = np.column_stack([obs_cx, obs_cy])
    if icao_pts is not None:
        obs_pts = np.vstack([obs_pts, icao_pts])
    sublog(f"  Total obstacle points (buildings + ICAO): {len(obs_pts)}")

    # Proximity gradient buffer (Hoa et al., 2021): 1 - min(d / 50m, 1.0)
    if len(obs_pts) > 0:
        dist_to_obs   = kdtree_dist(valid_pts, obs_pts, nrows, ncols, valid_idx)
        prox_risk_raw = np.clip(1 - dist_to_obs/PROX_THRESH, 0, 1).astype('float32')
        prox_risk     = np.where(bnd_mask, prox_risk_raw, np.nan).astype('float32')
    else:
        prox_risk = np.where(bnd_mask, 0.0, np.nan).astype('float32')
    sublog(f"  Proximity risk: mean={np.nanmean(prox_risk):.4f}, "
           f"std={np.nanstd(prox_risk):.4f}")

    # Composite cost map (AHP weighted sum)
    rail_safe = np.where(np.isnan(rail_risk), 0, rail_risk).astype('float32')
    cost_soft = (W_WIND * wind_risk +
                 W_PROX * prox_risk +
                 W_GNSS * gnss_risk +
                 W_RAIL * rail_safe)
    cost_soft = np.where(bnd_mask, cost_soft, np.nan).astype('float32')

    # Set very low cost at vertiport locations (landing/take-off nodes)
    if vp_gdf is not None:
        for _, row in vp_gdf.iterrows():
            c = int((row.geometry.x - minx) / RESOLUTION)
            r = int((maxy - row.geometry.y) / RESOLUTION)
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if 0 <= r+dr < nrows and 0 <= c+dc < ncols:
                        cost_soft[r+dr, c+dc] = 0.05

    valid_cost = cost_soft[bnd_mask & ~np.isnan(cost_soft)]
    sublog(f"  Cost map: mean={valid_cost.mean():.4f}, "
           f"std={valid_cost.std():.4f}, "
           f"max={valid_cost.max():.4f}")

    save_raster(cost_soft, transform_5m, TARGET_CRS, os.path.join(out_dir, 'cost_map_5m.tif'))
    save_raster(wind_risk, transform_5m, TARGET_CRS, os.path.join(out_dir, 'wind_risk_5m.tif'))
    save_raster(prox_risk, transform_5m, TARGET_CRS, os.path.join(out_dir, 'prox_risk_5m.tif'))
    save_raster(gnss_risk, transform_5m, TARGET_CRS, os.path.join(out_dir, 'gnss_risk_5m.tif'))
    save_raster(rail_risk, transform_5m, TARGET_CRS, os.path.join(out_dir, 'rail_risk_5m.tif'))
    sublog(f"  {alt} m layer saved")


# ============================================================
# Step 7: Quality checks
# ============================================================
log("Step 7: Quality checks")
w_sum = W_WIND + W_PROX + W_GNSS + W_RAIL
print(f"  AHP weight sum: {w_sum:.2f} {'OK' if abs(w_sum - 1.0) < 0.001 else 'ERROR'}")
print(f"  Wind source: {'HadUK-Grid' if wind_loaded else 'uniform proxy'}")

print(f"\n  Proximity risk by altitude layer:")
print(f"  {'Alt':>5} {'Obstacles':>10} {'Mean':>10} {'Std':>10}")
print(f"  {'-'*38}")
means = []
for alt in ALTITUDES:
    with rasterio.open(os.path.join(config.cost_dir(alt), 'prox_risk_5m.tif')) as src:
        prox = src.read(1).astype('float32')
        prox = np.where(prox == src.nodata, np.nan, prox)
    bld_obs = gpd.read_file(os.path.join(BLD_BASE, f'buildings_obstacle_{alt}m.shp'))
    v = prox[bnd_mask & ~np.isnan(prox)]
    print(f"  {alt:>4} m  {len(bld_obs):>9}  {v.mean():>9.4f}  {v.std():>9.4f}")
    means.append(v.mean())

monotone = all(means[i] >= means[i+1] for i in range(len(means)-1))
print(f"\n  Proximity monotonically decreasing: {'yes' if monotone else 'NO'}")
print(f"  {' > '.join(f'{m:.4f}' for m in means)}")

log("Phase 4 complete")
