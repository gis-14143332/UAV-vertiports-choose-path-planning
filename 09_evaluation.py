"""
09_evaluation.py
Full evaluation: answers RQ1, RQ2, and RQ3.

Analysis A -- Theta* vs straight-line risk comparison (RQ3 core)
Analysis B -- Four-layer statistical comparison
             Wilcoxon signed-rank + paired t-test + Cohen's d
Analysis C -- Composite scoring (risk 0.7, time 0.3; Saaty, 1980)
Analysis D -- Spatial analysis by urban morphological zone
Analysis E -- Network service coverage
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from scipy.stats import wilcoxon, ttest_rel
from shapely.geometry import Point
from shapely.ops import unary_union
from itertools import combinations

warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
import config

OUT_DIR   = config.ROUTES_DIR
EVAL_DIR  = config.EVAL_DIR
VERTIPORT = config.VERTIPORT_SHP
BOUNDARY  = config.BOUNDARY
COST_MAPS = {alt: os.path.join(config.cost_dir(alt), 'cost_map_5m.tif')
             for alt in [25, 50, 80, 120]}
os.makedirs(EVAL_DIR, exist_ok=True)

# ============================================================
# Parameters
# ============================================================
TARGET_CRS  = config.TARGET_CRS
RESOLUTION  = 5
ALTITUDES   = [25, 50, 80, 120]
W_RISK      = 0.7     # composite scoring: risk weight (Saaty, 1980)
W_TIME      = 0.3     # composite scoring: time weight
CITY_CENTRE = (383500, 398000)    # BNG; Manchester city centre
ZONE_INNER  = 2000    # City Centre: < 2 km from city centre
ZONE_MID    = 5000    # Mid-ring: 2-5 km; Southern: > 5 km

# ============================================================
# Utility
# ============================================================
def log(msg):    print(f"\n{'='*60}\n{msg}\n{'='*60}")
def sublog(msg): print(f"  -> {msg}")

def cohen_d(a, b):
    diff = np.array(a) - np.array(b)
    return diff.mean() / (diff.std() + 1e-10)


# ============================================================
# Load data
# ============================================================
log("Load data")
routes_df = pd.read_csv(os.path.join(OUT_DIR, 'all_routes_new.csv'))
sublog(f"Routes: {len(routes_df)} rows")

vp_gdf = gpd.read_file(VERTIPORT).to_crs(TARGET_CRS).reset_index(drop=True)
n_vp   = len(vp_gdf)
sublog(f"Vertiports: {n_vp}")

boundary   = gpd.read_file(BOUNDARY).to_crs(TARGET_CRS)
total_area = unary_union(boundary.geometry).area / 1e6
all_pairs  = list(combinations(range(n_vp), 2))

# Load cost arrays for straight-line sampling
cost_arrays     = {}
cost_transforms = {}
cost_shapes     = {}
for alt in ALTITUDES:
    with rasterio.open(COST_MAPS[alt]) as src:
        arr = src.read(1).astype('float32')
        arr = np.where(arr == src.nodata, np.nan, arr)
        cost_arrays[alt]     = arr
        cost_transforms[alt] = src.transform
        cost_shapes[alt]     = arr.shape

# Load edge path GeoPackages
edge_gdfs = {}
for alt in ALTITUDES:
    try:
        edge_gdfs[alt] = gpd.read_file(
            os.path.join(OUT_DIR, f'edge_paths_{alt}m.gpkg'))
    except:
        edge_gdfs[alt] = None


def w2g(x, y, alt):
    t = cost_transforms[alt]
    return (int((t.f - y) / RESOLUTION), int((x - t.c) / RESOLUTION))

def sample_straight_cost(xi, yi, xj, yj, alt):
    """Sample cost along straight line at RESOLUTION intervals."""
    cost = cost_arrays[alt]
    nrows, ncols = cost_shapes[alt]
    dist = np.sqrt((xj-xi)**2 + (yj-yi)**2)
    n_s  = max(int(dist / RESOLUTION), 2)
    vals = []
    for k in range(n_s):
        t = k / (n_s - 1)
        px = xi + t*(xj-xi); py = yi + t*(yj-yi)
        r, c = w2g(px, py, alt)
        if 0 <= r < nrows and 0 <= c < ncols:
            v = float(cost[r, c])
            if not np.isnan(v): vals.append(v)
    return float(np.mean(vals)) if vals else 0.3


# ============================================================
# Analysis A: Theta* vs straight-line (RQ3)
# ============================================================
log("Analysis A: Theta* vs straight-line risk (RQ3)")
rq3_rows = []
for alt in ALTITUDES:
    gdf = edge_gdfs[alt]
    if gdf is None: continue
    for _, row in gdf.iterrows():
        vi = int(row['vi']) - 1
        vj = int(row['vj']) - 1
        xi, yi = vp_gdf.iloc[vi].geometry.x, vp_gdf.iloc[vi].geometry.y
        xj, yj = vp_gdf.iloc[vj].geometry.x, vp_gdf.iloc[vj].geometry.y

        theta_cost    = float(row['mean_cost'])
        theta_length  = float(row['path_len_m'])
        edge_dist     = float(row['edge_dist_m'])
        direct_los    = bool(row['direct_los'])
        straight_cost = sample_straight_cost(xi, yi, xj, yj, alt)
        risk_improve  = ((straight_cost - theta_cost) / straight_cost * 100
                         if straight_cost > 0 else 0)
        detour_ratio  = theta_length / edge_dist if edge_dist > 0 else 1.0

        rq3_rows.append({
            'altitude_m'         : alt,
            'vi': vi+1, 'vj': vj+1,
            'edge_dist_m'        : edge_dist,
            'theta_length_m'     : theta_length,
            'theta_mean_cost'    : round(theta_cost, 4),
            'straight_mean_cost' : round(straight_cost, 4),
            'risk_improvement_pct': round(risk_improve, 2),
            'detour_ratio'       : round(detour_ratio, 3),
            'direct_los_blocked' : not direct_los,
            'n_waypoints'        : int(row['n_waypoints']),
        })

rq3_df = pd.DataFrame(rq3_rows)
rq3_df.to_csv(os.path.join(EVAL_DIR, 'rq3_risk_comparison.csv'),
              index=False, encoding='utf-8')
sublog(f"RQ3 data: {len(rq3_df)} rows")


# ============================================================
# Analysis B: Statistical comparison across altitude layers
# ============================================================
log("Analysis B: Statistical tests")

# Build paired cost arrays (one value per OD pair per altitude)
paired = {alt: [] for alt in ALTITUDES}
for vi, vj in all_pairs:
    rows = {alt: routes_df[(routes_df['altitude_m'] == alt) &
                            (routes_df['vi'] == vi+1) &
                            (routes_df['vj'] == vj+1)]
            for alt in ALTITUDES}
    if all(len(r) > 0 for r in rows.values()):
        for alt in ALTITUDES:
            paired[alt].append(float(rows[alt]['total_cost'].values[0]))

n_paired = len(paired[ALTITUDES[0]])
sublog(f"Paired sample size: {n_paired}")

# Descriptive statistics
alt_stats = []
for alt in ALTITUDES:
    c = routes_df[routes_df['altitude_m'] == alt]['total_cost'].values
    t = routes_df[routes_df['altitude_m'] == alt]['total_time_min'].values
    alt_stats.append({
        'altitude_m': alt,
        'cost_mean': round(c.mean(), 4), 'cost_std': round(c.std(), 4),
        'cost_median': round(np.median(c), 4),
        'cost_p25': round(np.percentile(c, 25), 4),
        'cost_p75': round(np.percentile(c, 75), 4),
        'time_mean_min': round(t.mean(), 2), 'time_std_min': round(t.std(), 2),
    })
pd.DataFrame(alt_stats).to_csv(
    os.path.join(EVAL_DIR, 'altitude_stats.csv'), index=False, encoding='utf-8')

# Pairwise significance tests
test_rows = []
for a, b in [(25,50),(25,80),(25,120),(50,80),(50,120),(80,120)]:
    if len(paired[a]) < 5: continue
    w_stat, w_p = wilcoxon(paired[a], paired[b])
    t_stat, t_p = ttest_rel(paired[a], paired[b])
    d           = cohen_d(paired[a], paired[b])
    test_rows.append({
        'alt_a': a, 'alt_b': b,
        'wilcoxon_stat': round(w_stat, 2), 'wilcoxon_p': round(w_p, 4),
        'ttest_t': round(t_stat, 3),       'ttest_p': round(t_p, 4),
        'cohens_d': round(d, 3),
        'significant': w_p < 0.05,
    })
test_df = pd.DataFrame(test_rows)
test_df.to_csv(os.path.join(EVAL_DIR, 'statistical_tests.csv'),
               index=False, encoding='utf-8')
sublog(f"Statistical tests: {len(test_df)} pairs")

# Optimal altitude distribution (pure risk)
best_counts = {alt: 0 for alt in ALTITUDES}
for i in range(n_paired):
    best = min(ALTITUDES, key=lambda a: paired[a][i])
    best_counts[best] += 1


# ============================================================
# Analysis C: Composite scoring (risk 0.7, time 0.3)
# ============================================================
log("Analysis C: Composite scoring")
all_c = routes_df['total_cost'].values
all_t = routes_df['total_time_min'].values
c_min, c_max = all_c.min(), all_c.max()
t_min, t_max = all_t.min(), all_t.max()

def norm(x, mn, mx):
    return (x - mn) / (mx - mn) if mx > mn else 0.0

comp_rows = []
for vi, vj in all_pairs:
    scores = {}
    for alt in ALTITUDES:
        r = routes_df[(routes_df['altitude_m'] == alt) &
                      (routes_df['vi'] == vi+1) &
                      (routes_df['vj'] == vj+1)]
        if len(r) == 0: continue
        c = float(r['total_cost'].values[0])
        t = float(r['total_time_min'].values[0])
        scores[alt] = W_RISK * norm(c, c_min, c_max) + W_TIME * norm(t, t_min, t_max)
    if len(scores) == len(ALTITUDES):
        best = min(scores, key=scores.get)
        comp_rows.append({'vi': vi+1, 'vj': vj+1,
                          **{f'score_{a}m': round(scores[a], 4) for a in ALTITUDES},
                          'best_alt_comp': best,
                          'best_score': round(scores[best], 4)})
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(os.path.join(EVAL_DIR, 'composite_score.csv'),
               index=False, encoding='utf-8')
sublog(f"Composite scores: {len(comp_df)} OD pairs")


# ============================================================
# Analysis D: Spatial analysis by morphological zone
# City Centre < 2 km, Mid-ring 2-5 km, Southern > 5 km
# ============================================================
log("Analysis D: Spatial analysis by morphological zone")
cx, cy = CITY_CENTRE
spatial_rows = []
for vi, vj in all_pairs:
    xi, yi = vp_gdf.iloc[vi].geometry.x, vp_gdf.iloc[vi].geometry.y
    xj, yj = vp_gdf.iloc[vj].geometry.x, vp_gdf.iloc[vj].geometry.y
    mid_x  = (xi + xj) / 2; mid_y = (yi + yj) / 2
    dist_c = np.sqrt((mid_x - cx)**2 + (mid_y - cy)**2) / 1000

    if dist_c > ZONE_MID/1000:
        zone = 'Southern'
    elif dist_c < ZONE_INNER/1000:
        zone = 'City Centre'
    else:
        zone = 'Mid-ring'

    row_d = {'vi': vi+1, 'vj': vj+1,
             'dist_centre_km': round(dist_c, 2), 'zone': zone}
    for alt in ALTITUDES:
        r = routes_df[(routes_df['altitude_m'] == alt) &
                      (routes_df['vi'] == vi+1) &
                      (routes_df['vj'] == vj+1)]
        if len(r) > 0:
            row_d[f'cost_{alt}m'] = float(r['total_cost'].values[0])
            row_d[f'time_{alt}m'] = float(r['total_time_min'].values[0])
            row_d[f'hops_{alt}m'] = int(r['hops'].values[0])
    spatial_rows.append(row_d)

spatial_df = pd.DataFrame(spatial_rows)
spatial_df.to_csv(os.path.join(EVAL_DIR, 'spatial_analysis.csv'),
                  index=False, encoding='utf-8')

geoms_sp = [Point((vp_gdf.iloc[vi].geometry.x + vp_gdf.iloc[vj].geometry.x)/2,
                   (vp_gdf.iloc[vi].geometry.y + vp_gdf.iloc[vj].geometry.y)/2)
            for vi, vj in all_pairs]
gpd.GeoDataFrame(spatial_df, geometry=geoms_sp, crs=TARGET_CRS).to_file(
    os.path.join(EVAL_DIR, 'spatial_analysis.shp'))
sublog(f"Spatial analysis: {len(spatial_df)} OD pairs")


# ============================================================
# Analysis E: Network service coverage
# ============================================================
log("Analysis E: Service coverage")
circles  = [Point(row.geometry.x, row.geometry.y).buffer(1500)
            for _, row in vp_gdf.iterrows()]
coverage = unary_union(circles).intersection(unary_union(boundary.geometry))
area_cov = coverage.area / 1e6
sublog(f"Study area: {total_area:.2f} km2")
sublog(f"Service coverage (1.5 km): {area_cov:.2f} km2 "
       f"({area_cov/total_area*100:.1f}%)")


# ============================================================
# Print summary
# ============================================================
log("Summary")
print(f"\n  RQ3 -- Theta* vs straight-line:")
for alt in ALTITUDES:
    sub = rq3_df[rq3_df['altitude_m'] == alt]
    print(f"    {alt} m: improved={len(sub[sub['risk_improvement_pct']>0])}/{len(sub)} "
          f"mean_improve={sub['risk_improvement_pct'].mean():.2f}% "
          f"mean_detour={sub['detour_ratio'].mean():.3f}x "
          f"blocked={sub['direct_los_blocked'].sum()}")

print(f"\n  Statistical tests:")
for _, row in test_df.iterrows():
    mag = 'large' if abs(row['cohens_d']) > 0.8 else ('medium' if abs(row['cohens_d']) > 0.5 else 'small')
    print(f"    {int(row['alt_a'])} vs {int(row['alt_b'])} m: "
          f"W-p={row['wilcoxon_p']:.4f} d={row['cohens_d']:.3f} ({mag})")

print(f"\n  Optimal altitude (pure risk):")
for alt in ALTITUDES:
    n = best_counts[alt]
    print(f"    {alt} m: {n} OD pairs ({n/n_paired*100:.1f}%)")

print(f"\n  Composite optimal altitude:")
for alt, n in comp_df['best_alt_comp'].value_counts().items():
    print(f"    {alt} m: {n} ({n/len(comp_df)*100:.1f}%)")

print(f"\n  Spatial zones:")
for zone in ['City Centre', 'Mid-ring', 'Southern']:
    sub = spatial_df[spatial_df['zone'] == zone]
    bc  = {}
    for _, row in sub.dropna(subset=[f'cost_{a}m' for a in ALTITUDES]).iterrows():
        best = min(ALTITUDES, key=lambda a: row[f'cost_{a}m'])
        bc[best] = bc.get(best, 0) + 1
    dominant = max(bc, key=bc.get) if bc else 'N/A'
    print(f"    {zone} (n={len(sub)}): dominant={dominant} m  {bc}")

log("Phase 6 complete")
