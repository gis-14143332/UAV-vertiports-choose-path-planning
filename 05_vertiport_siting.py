"""
05_vertiport_siting.py
Capacitated MCLP with dual-radius formulation and network connectivity constraint.
Identifies the optimal number of vertiports (p) for UAV food delivery in Manchester.

Key design decisions:
  - Siting radius r1 = 1.0 km: anchors vertiports within restaurant clusters
  - Service radius r2 = 1.5 km: assesses delivery coverage (Church, 1979)
  - Capacity C = 100 (P25 of CDI distribution): forces demand-concentrated placement
  - Connectivity constraint 3.0 km: ensures full network routing capability
  - Stopping criterion: marginal gain < 2% (Church & ReVelle, 1974)

References:
  Church & ReVelle (1974) Papers of the Regional Science Association
  Church (1979) Geographical Analysis
  Malczewski (1999) GIS and Multicriteria Decision Analysis
  Saaty (1980) The Analytic Hierarchy Process
"""

import os, warnings, zipfile, tempfile, re
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from shapely.geometry import Point
from shapely.ops import unary_union
import rasterio
import fiona

warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
import config

CDI_TIF    = config.CDI_TIF
BOUNDARY   = config.BOUNDARY
# Siting used the first OS Building Heights download only (tiles sd/sj), as in the original run.
BH_BASE    = config.BH_DIRS[0]
OSM_ROADS  = config.OSM_ROADS
CAA_KMZ    = config.CAA_KMZ
ICAO_XLS   = config.ICAO_XLS
RESTAURANT = config.RESTAURANTS_CSV
OUT_DIR    = config.VERTIPORT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Parameters
# ============================================================
TARGET_CRS        = 'EPSG:27700'
CANDIDATE_SPACING = 200       # candidate grid spacing (m)
BH_CLEARANCE      = 45        # max building height within candidate radius (m); CAA CAP722
BH_RADIUS         = 200       # radius for building clearance check (m)
NETWORK_RADIUS    = 3000      # connectivity constraint threshold (m); Meituan operational range
MARGINAL_THRESH   = 2.0       # stopping criterion: marginal gain < 2% (Church & ReVelle, 1974)
CAPACITIES        = [100, 200, 400]   # low / medium / high fixed capacity (CDI weight units)
CAPACITY_LABELS   = ['low', 'mid', 'high']
SITING_RADIUS     = 1000      # MCLP siting radius (m)
SERVICE_RADIUS    = 1500      # service coverage radius (m); Meituan last-mile delivery range
MCLP_RADII        = [1000, 1500, 3000]  # radii for sensitivity analysis
JENKS_CLASSES     = 4
JENKS_SCORES      = [1.0, 0.7, 0.4, 0.1]

# ============================================================
# Utility functions
# ============================================================
def log(msg):    print(f"\n{'='*60}\n{msg}\n{'='*60}")
def sublog(msg): print(f"  -> {msg}")


def jenks_breaks(values, n_classes):
    """Compute Jenks natural breaks (Jenks & Caspall, 1971)."""
    vals = np.sort(np.array(values))
    n    = len(vals)
    if n <= n_classes:
        return vals.tolist()
    mat1 = np.zeros((n+1, n_classes+1))
    mat2 = np.full((n+1, n_classes+1), np.inf)
    mat2[1, 1:] = 0
    for l in range(2, n+1):
        s1 = s2 = w = 0.0
        for m in range(1, l+1):
            i3 = l - m + 1
            val = vals[i3 - 1]
            w += 1; s2 += val*val; s1 += val
            v = s2 - s1*s1/w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, n_classes+1):
                    if mat2[l, j] >= v + mat2[i4, j-1]:
                        mat1[l, j] = i3
                        mat2[l, j] = v + mat2[i4, j-1]
        if l == 1:
            mat1[l, 1] = 1
            mat2[l, 1] = 0
    breaks = [vals[0]]
    k = n
    for j in range(n_classes, 1, -1):
        idx = int(mat1[k, j]) - 1
        breaks.insert(1, vals[idx])
        k = int(mat1[k, j]) - 1
    breaks.append(vals[-1])
    return breaks


def jenks_score(values, n_classes=4, scores=None):
    """Assign ordinal scores based on Jenks classes."""
    if scores is None:
        scores = JENKS_SCORES
    breaks = jenks_breaks(values, n_classes)
    result = np.zeros(len(values))
    for i, v in enumerate(values):
        for cls in range(n_classes):
            if v <= breaks[cls+1]:
                result[i] = scores[cls]
                break
        else:
            result[i] = scores[-1]
    return result


def minmax_norm(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def is_network_connected(selected, cand_coords, max_dist=3000):
    """BFS check: are all selected vertiports reachable within max_dist?"""
    if len(selected) <= 1:
        return True
    n   = len(selected)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = cand_coords[selected[i]]
            xj, yj = cand_coords[selected[j]]
            if np.sqrt((xi-xj)**2 + (yi-yj)**2) <= max_dist:
                adj[i].append(j)
                adj[j].append(i)
    visited = {0}
    queue   = [0]
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == n


# ============================================================
# Step 1: Load boundary
# ============================================================
log("Step 1: Load boundary")
boundary       = gpd.read_file(BOUNDARY).to_crs(TARGET_CRS)
boundary_union = unary_union(boundary.geometry)
bounds         = boundary.total_bounds
minx, miny, maxx, maxy = bounds


# ============================================================
# Step 2: OS Building Heights
# ============================================================
log("Step 2: Load OS Building Heights")
bh_gdfs = []
for folder in ['sd', 'sj']:
    fp = os.path.join(BH_BASE, folder)
    for item in sorted(os.listdir(fp)):
        if item.endswith('.gdb'):
            layers   = fiona.listlayers(os.path.join(fp, item))
            gdf_tile = gpd.read_file(os.path.join(fp, item), layer=layers[0])
            bh_gdfs.append(gdf_tile)

bh_all     = gpd.GeoDataFrame(pd.concat(bh_gdfs, ignore_index=True), crs=TARGET_CRS)
bh_all     = gpd.clip(bh_all, boundary)
bh_coords  = np.column_stack([bh_all.geometry.centroid.x,
                               bh_all.geometry.centroid.y])
bh_heights = bh_all['relhmax'].values
bh_tree    = cKDTree(bh_coords)
sublog(f"Buildings: {len(bh_all)}")


# ============================================================
# Step 3: OSM road network
# ============================================================
log("Step 3: Load OSM roads")
roads = gpd.read_file(OSM_ROADS).to_crs(TARGET_CRS)
main_road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
                   'residential', 'unclassified', 'motorway_link',
                   'trunk_link', 'primary_link']
if 'fclass' in roads.columns:
    roads = roads[roads['fclass'].isin(main_road_types)]
roads     = gpd.clip(roads, boundary)
road_pts  = []
for geom in roads.geometry:
    if geom is not None:
        try:   road_pts.extend(list(geom.coords))
        except: pass
road_pts  = np.array(road_pts)
road_tree = cKDTree(road_pts)
sublog(f"Road nodes: {len(road_pts)}")


# ============================================================
# Step 4: Restaurant demand data
# ============================================================
log("Step 4: Load restaurant demand")
rest_df  = pd.read_csv(RESTAURANT, encoding='utf-8')
rest_df  = rest_df[rest_df['demand_weight'].notna()].copy()
rest_gdf = gpd.GeoDataFrame(
    rest_df,
    geometry=gpd.points_from_xy(rest_df['final_lon'], rest_df['final_lat']),
    crs='EPSG:4326').to_crs(TARGET_CRS)
rest_gdf    = gpd.clip(rest_gdf, boundary)
rest_coords = np.column_stack([rest_gdf.geometry.x.values,
                                rest_gdf.geometry.y.values])
rest_tree   = cKDTree(rest_coords)
sublog(f"Restaurants: {len(rest_gdf)}")


# ============================================================
# Step 5: Hard constraints (CAA no-fly zones + ICAO obstacles)
# ============================================================
log("Step 5: Load hard constraints")

caa_buffer_geom  = None
icao_buffer_geom = None

# CAA no-fly zones
try:
    fiona.drvsupport.supported_drivers['KML']    = 'rw'
    fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(CAA_KMZ, 'r') as z:
            z.extractall(tmpdir)
        kml_files = [f for f in os.listdir(tmpdir) if f.lower().endswith('.kml')]
        if kml_files:
            kml_path = os.path.join(tmpdir, kml_files[0])
            for driver in ['KML', 'LIBKML']:
                try:
                    caa = gpd.read_file(kml_path, driver=driver)
                    break
                except: continue
            caa = caa.to_crs(TARGET_CRS)
            caa = gpd.clip(caa, boundary)
            if len(caa) > 0:
                caa_buffer_geom = unary_union(caa.geometry)
                sublog(f"CAA no-fly zones: {len(caa)}")
            else:
                sublog("No CAA no-fly zones within Manchester")
except Exception as e:
    sublog(f"CAA load skipped: {e}")


def parse_icao_coord(coord_str):
    """Parse ICAO DMS coordinate strings to decimal degrees."""
    coord_str = str(coord_str).replace('\r\n', ' ').strip()
    parts = coord_str.split()
    if len(parts) < 2:
        return None, None
    def dms(s):
        hemi = s[-1]; nums = s[:-1]
        if '.' in nums:
            int_p, dec_p = nums.split('.')
            s_dec = float('0.' + dec_p)
        else:
            int_p = nums; s_dec = 0.0
        L = len(int_p)
        if L == 5:   d, m, sc = int(int_p[0]),   int(int_p[1:3]), int(int_p[3:5]) + s_dec
        elif L == 6: d, m, sc = int(int_p[0:2]), int(int_p[2:4]), int(int_p[4:6]) + s_dec
        elif L == 7: d, m, sc = int(int_p[0:3]), int(int_p[3:5]), int(int_p[5:7]) + s_dec
        else:        d, m, sc = int(int_p[0:2]), int(int_p[2:4]), float(int_p[4:])  + s_dec
        dd = d + m/60 + sc/3600
        if hemi in ('S', 'W'): dd = -dd
        return dd
    try:
        return dms(parts[0]), dms(parts[1])
    except:
        return None, None


# ICAO obstacles (50 m exclusion buffer; ICAO Doc 10019)
try:
    icao_df  = pd.read_excel(ICAO_XLS, sheet_name='All')
    coords_p = icao_df['Obstacle Position'].apply(
        lambda x: pd.Series(parse_icao_coord(x), index=['lat', 'lon']))
    icao_df  = pd.concat([icao_df, coords_p], axis=1).dropna(subset=['lat', 'lon'])

    def parse_height(h):
        if pd.isna(h): return 0
        m = re.search(r'([\d.]+)\s*FT', str(h), re.IGNORECASE)
        if m: return float(m.group(1)) * 0.3048
        m2 = re.search(r'([\d.]+)\s*M', str(h), re.IGNORECASE)
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
        icao_buffer_geom = unary_union(icao_gdf.geometry.buffer(50))
        sublog(f"ICAO obstacles: {len(icao_gdf)}, 50 m buffer applied")
except Exception as e:
    sublog(f"ICAO load skipped: {e}")


# ============================================================
# Step 6: Generate candidate grid and apply hard constraints
# ============================================================
log("Step 6: Generate candidate grid")
xs = np.arange(minx + CANDIDATE_SPACING/2, maxx, CANDIDATE_SPACING)
ys = np.arange(miny + CANDIDATE_SPACING/2, maxy, CANDIDATE_SPACING)
xx, yy = np.meshgrid(xs, ys)
pts     = np.column_stack([xx.ravel(), yy.ravel()])

cand_gdf = gpd.GeoDataFrame(
    geometry=[Point(x, y) for x, y in pts], crs=TARGET_CRS)
cand_gdf = cand_gdf[
    cand_gdf.geometry.within(boundary_union)
].copy().reset_index(drop=True)
sublog(f"Within boundary: {len(cand_gdf)}")

if caa_buffer_geom is not None:
    cand_gdf = cand_gdf[~cand_gdf.geometry.within(caa_buffer_geom)].reset_index(drop=True)
if icao_buffer_geom is not None:
    cand_gdf = cand_gdf[~cand_gdf.geometry.within(icao_buffer_geom)].reset_index(drop=True)
    sublog(f"After ICAO exclusion: {len(cand_gdf)}")

# Exclude candidates near tall buildings (CAA CAP722 take-off clearance)
cx  = cand_gdf.geometry.x.values
cy  = cand_gdf.geometry.y.values
cand_coords_all = np.column_stack([cx, cy])
idx_list = bh_tree.query_ball_point(cand_coords_all, BH_RADIUS)
max_bh   = np.array([bh_heights[idx].max() if len(idx) > 0 else 0.0
                     for idx in idx_list])
cand_gdf = cand_gdf[max_bh <= BH_CLEARANCE].copy().reset_index(drop=True)
sublog(f"After building clearance filter: {len(cand_gdf)}")

cx          = cand_gdf.geometry.x.values
cy          = cand_gdf.geometry.y.values
cand_coords = np.column_stack([cx, cy])
n_cand      = len(cand_gdf)


# ============================================================
# Step 7: MCDA suitability scoring (three equal-weight criteria)
# CDI demand, road accessibility, building clearance (Malczewski, 1999; Saaty, 1980)
# ============================================================
log("Step 7: MCDA suitability scoring")

with rasterio.open(CDI_TIF) as src:
    cdi_samples = np.array([v[0] for v in src.sample(list(zip(cx, cy)))])
cdi_samples = np.where(np.isnan(cdi_samples) | (cdi_samples < 0), 0, cdi_samples)
score_cdi   = minmax_norm(cdi_samples)
cand_gdf['cdi_raw']   = cdi_samples
cand_gdf['score_cdi'] = score_cdi

road_dists, _ = road_tree.query(cand_coords)
score_road    = jenks_score(road_dists)
cand_gdf['road_dist_m'] = road_dists
cand_gdf['score_road']  = score_road

idx_list2 = bh_tree.query_ball_point(cand_coords, BH_RADIUS)
mean_bh   = np.array([bh_heights[idx].mean() if len(idx) > 0 else 0.0
                      for idx in idx_list2])
score_bh  = jenks_score(mean_bh)
cand_gdf['mean_bh_m'] = mean_bh
cand_gdf['score_bh']  = score_bh

W = 1/3  # equal weights
cand_gdf['mcda_score'] = W*score_cdi + W*score_road + W*score_bh
cand_gdf.to_file(os.path.join(OUT_DIR, 'vertiport_candidates.shp'))
sublog(f"MCDA score: mean={cand_gdf['mcda_score'].mean():.4f}")


# ============================================================
# Step 8: Demand surface (CDI raster sampled at 100 m)
# ============================================================
log("Step 8: Load demand surface")
with rasterio.open(CDI_TIF) as src:
    cdi_arr   = src.read(1).astype('float64')
    nodata    = src.nodata
    transform = src.transform
cdi_arr = np.where(cdi_arr == nodata, np.nan, cdi_arr)

step = 2
dr, dc     = np.where(~np.isnan(cdi_arr[::step, ::step]))
dr, dc     = dr*step, dc*step
demand_x   = transform.c + dc*transform.a + transform.a/2
demand_y   = transform.f + dr*transform.e + transform.e/2
demand_w   = cdi_arr[dr, dc]
demand_coords = np.column_stack([demand_x, demand_y])
total_demand  = demand_w.sum()
sublog(f"Demand cells: {len(demand_x)}, total weight: {total_demand:.2f}")


# ============================================================
# Step 9: Capacitated MCLP with connectivity constraint
# ============================================================
log("Step 9: Capacitated MCLP")

def run_mclp(radius, label, capacity=200):
    """
    Greedy Capacitated MCLP with network connectivity constraint.
    Stops when marginal coverage gain < MARGINAL_THRESH (2%).
    Capacity C is fixed (not demand-proportional).
    """
    sublog(f"\n-- {label} (radius {radius/1000:.1f} km, C={capacity}) --")
    demand_tree = cKDTree(demand_coords)
    covered_by  = demand_tree.query_ball_point(cand_coords, radius)

    selected       = []
    covered        = set()
    coverage_curve = []
    prev_rate      = 0.0
    p              = 0
    no_progress    = 0

    while True:
        p    += 1
        best_gain = -1
        best_i    = -1

        for i, idxs in enumerate(covered_by):
            if i in selected:
                continue
            # Connectivity constraint: must be within NETWORK_RADIUS of existing network
            if len(selected) > 0:
                xi, yi = cand_coords[i]
                connected = any(
                    np.sqrt((xi - cand_coords[s][0])**2 +
                            (yi - cand_coords[s][1])**2) <= NETWORK_RADIUS
                    for s in selected)
                if not connected:
                    continue
            # Capacity-constrained gain: sort by weight, take up to C
            new_idxs = list(set(idxs) - covered)
            if not new_idxs:
                continue
            new_w    = demand_w[new_idxs]
            sort_idx = np.argsort(new_w)[::-1]
            cum, gain = 0.0, 0.0
            for j in sort_idx:
                if cum + new_w[j] > capacity:
                    break
                cum  += new_w[j]
                gain += new_w[j]
            if gain > best_gain:
                best_gain = gain
                best_i    = i

        if best_i == -1:
            sublog(f"  p={p}: no candidates available, stopping")
            break

        new_idxs = list(set(covered_by[best_i]) - covered)
        new_w    = demand_w[new_idxs]
        sort_idx = np.argsort(new_w)[::-1]
        cum, actually_covered = 0.0, []
        for j in sort_idx:
            if cum + new_w[j] > capacity:
                break
            cum += new_w[j]
            actually_covered.append(new_idxs[j])

        selected.append(best_i)
        covered.update(actually_covered)

        if not is_network_connected(selected, cand_coords, NETWORK_RADIUS):
            selected.pop()
            covered.difference_update(actually_covered)
            no_progress += 1
            if no_progress > 10:
                sublog(f"  {no_progress} consecutive connectivity failures, stopping")
                break
            p -= 1
            covered_by[best_i] = []  # temporarily exclude this candidate
            continue

        no_progress = 0
        rate     = demand_w[list(covered)].sum() / total_demand * 100
        marginal = rate - prev_rate
        coverage_curve.append(rate)
        prev_rate = rate
        sublog(f"  p={p}: coverage={rate:.1f}%, marginal={marginal:+.1f}%")

        if marginal < MARGINAL_THRESH and p >= 5:
            sublog(f"  Marginal gain {marginal:.1f}% < {MARGINAL_THRESH}%, optimal p={p}")
            break

    return selected, coverage_curve


# Primary analysis: three capacity scenarios at 1.0 km siting radius
mclp_results = {}
for capacity, cap_label in zip(CAPACITIES, CAPACITY_LABELS):
    sel, curve = run_mclp(SITING_RADIUS, f"{cap_label}_1km", capacity)
    # Re-assess coverage at 1.5 km service radius (dual-radius model)
    demand_tree_svc = cKDTree(demand_coords)
    covered_svc     = set()
    for sc in cand_coords[sel]:
        covered_svc.update(demand_tree_svc.query_ball_point(sc, SERVICE_RADIUS))
    service_cov = demand_w[list(covered_svc)].sum() / total_demand * 100
    mclp_results[(cap_label, SITING_RADIUS)] = {
        'selected'        : sel,
        'coverage_curve'  : curve,
        'optimal_p'       : len(sel),
        'siting_coverage' : curve[-1] if curve else 0,
        'service_coverage': service_cov,
        'capacity'        : capacity,
    }

# Sensitivity analysis: three service radii at medium capacity
for radius in MCLP_RADII:
    sel, curve = run_mclp(radius, f"sensitivity_{radius/1000:.0f}km", CAPACITIES[1])
    mclp_results[('sensitivity', radius)] = {
        'selected'      : sel,
        'coverage_curve': curve,
        'optimal_p'     : len(sel),
        'final_coverage': curve[-1] if curve else 0,
    }


# ============================================================
# Step 10: Save primary solution (low capacity, dual-radius)
# ============================================================
log("Step 10: Save primary solution")
main_result = mclp_results[('low', SITING_RADIUS)]
sel_main    = main_result['selected']
sel_gdf     = cand_gdf.iloc[sel_main].copy()
sel_gdf['rank']             = range(1, len(sel_gdf)+1)
sel_gdf['siting_radius_m']  = SITING_RADIUS
sel_gdf['service_radius_m'] = SERVICE_RADIUS

fname = os.path.basename(config.VERTIPORT_SHP)   # fixed name; the count p is logged below
sel_gdf.to_file(config.VERTIPORT_SHP)
sublog(f"Saved: {fname} ({len(sel_main)} vertiports)")


# ============================================================
# Step 11: Evaluation report
# ============================================================
log("Step 11: Evaluation report")
vlines = []
def vlog(msg): print(f"  {msg}"); vlines.append(msg)

vlog("="*60)
vlog("Phase 3 Capacitated MCLP -- Evaluation Report")
vlog(f"Capacities: {CAPACITIES} (low/mid/high)")
vlog(f"Connectivity threshold: {NETWORK_RADIUS/1000:.1f} km")
vlog(f"Stopping criterion: marginal gain < {MARGINAL_THRESH}%")
vlog("="*60)

# Per-scenario summary
vlog(f"\n[A] Dual-radius results (siting {SITING_RADIUS/1000:.1f} km, service {SERVICE_RADIUS/1000:.1f} km)")
vlog(f"  {'Label':>6} {'C':>5} {'p':>5} {'Siting%':>8} {'Service%':>9} {'CDI mult':>9} {'Connected':>10}")
vlog(f"  {'-'*55}")
for cap_label, capacity in zip(CAPACITY_LABELS, CAPACITIES):
    r   = mclp_results[(cap_label, SITING_RADIUS)]
    sel = r['selected']
    cdi_mult = (cand_gdf.iloc[sel]['cdi_raw'].values.mean() /
                cand_gdf['cdi_raw'].mean())
    ok = is_network_connected(sel, cand_coords, NETWORK_RADIUS)
    vlog(f"  {cap_label:>6} {capacity:>5} {r['optimal_p']:>5}  "
         f"{r['siting_coverage']:>7.1f}%  {r['service_coverage']:>8.1f}%  "
         f"{cdi_mult:>8.2f}x  {'yes' if ok else 'NO'}")

# Sensitivity analysis
vlog(f"\n[B] Service radius sensitivity (C={CAPACITIES[1]})")
vlog(f"  {'Radius':>7} {'p':>5} {'Coverage%':>10} {'CDI mult':>9}")
vlog(f"  {'-'*35}")
for radius in MCLP_RADII:
    r   = mclp_results[('sensitivity', radius)]
    sel = r['selected']
    cdi_mult = (cand_gdf.iloc[sel]['cdi_raw'].values.mean() /
                cand_gdf['cdi_raw'].mean())
    vlog(f"  {radius/1000:.1f} km  {r['optimal_p']:>5}  "
         f"{r['final_coverage']:>9.1f}%  {cdi_mult:>8.2f}x")

# Primary solution detail
r        = mclp_results[('low', SITING_RADIUS)]
sel      = r['selected']
sel_cdi  = cand_gdf.iloc[sel]['cdi_raw'].values
sel_coord= cand_coords[sel]

vlog(f"\n[C] Primary solution detail (low capacity, dual-radius)")
vlog(f"  Optimal p: {r['optimal_p']}")
vlog(f"  Siting coverage (1.0 km):  {r['siting_coverage']:.1f}%")
vlog(f"  Service coverage (1.5 km): {r['service_coverage']:.1f}%")
vlog(f"  CDI multiplier: {sel_cdi.mean()/cand_gdf['cdi_raw'].mean():.2f}x candidate-pool mean")

if len(sel_coord) >= 2:
    dists = pdist(sel_coord)
    vlog(f"  Min inter-vertiport distance: {dists.min():.0f} m")
    vlog(f"  Mean inter-vertiport distance: {dists.mean():.0f} m")

rest_dists, _ = rest_tree.query(sel_coord)
np.random.seed(42)
rand_idx      = np.random.choice(n_cand, len(sel), replace=False)
rand_dists, _ = rest_tree.query(cand_coords[rand_idx])
vlog(f"  Mean distance to nearest restaurant: {rest_dists.mean():.0f} m")
vlog(f"  Random baseline: {rand_dists.mean():.0f} m")
vlog(f"  Improvement: {(1 - rest_dists.mean()/rand_dists.mean())*100:.1f}%")

# Connectivity check
connected     = is_network_connected(sel, cand_coords, NETWORK_RADIUS)
nb_counts     = []
for vi in sel:
    xi, yi = cand_coords[vi]
    n_nb = sum(1 for vj in sel if vj != vi and
               np.sqrt((xi-cand_coords[vj][0])**2 +
                       (yi-cand_coords[vj][1])**2) <= NETWORK_RADIUS)
    nb_counts.append(n_nb)
vlog(f"  Network connected: {'yes' if connected else 'NO'}")
vlog(f"  Mean neighbours per vertiport: {np.mean(nb_counts):.1f}")
vlog(f"  Min neighbours: {min(nb_counts)}")

with open(os.path.join(OUT_DIR, 'mclp_report.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(vlines))
sublog("Report: mclp_report.txt")


# ============================================================
# Step 12: Save coverage curve CSV
# ============================================================
curve_data = {}
for cap_label in CAPACITY_LABELS:
    r = mclp_results[(cap_label, SITING_RADIUS)]
    curve_data[cap_label] = pd.Series(
        r['coverage_curve'],
        index=range(1, len(r['coverage_curve'])+1))
for radius in MCLP_RADII:
    r = mclp_results[('sensitivity', radius)]
    curve_data[f'sensitivity_{radius/1000:.1f}km_mid'] = pd.Series(
        r['coverage_curve'],
        index=range(1, len(r['coverage_curve'])+1))

pd.DataFrame(curve_data).to_csv(
    config.CURVES_CSV,
    index_label='p', encoding='utf-8')
sublog("Coverage curves: coverage_curves.csv")

log("Phase 3 complete")
