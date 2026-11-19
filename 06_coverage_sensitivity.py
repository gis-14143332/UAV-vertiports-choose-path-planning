"""
06_coverage_sensitivity.py
Supplement coverage_curves.csv with 1.5 km and 3.0 km service radius curves
(medium capacity C = 200) for sensitivity analysis figures.
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import rasterio

warnings.filterwarnings('ignore')

import config

CDI_TIF    = config.CDI_TIF
BOUNDARY   = config.BOUNDARY
CANDIDATES = config.CANDIDATES_SHP

TARGET_CRS     = 'EPSG:27700'
RADII          = [1500, 3000]
CAPACITY_MID   = 200
NETWORK_RADIUS = 3000
MARGINAL_THRESH= 2.0
MAX_P          = 20

def log(msg):    print(f"\n{'='*50}\n{msg}\n{'='*50}")
def sublog(msg): print(f"  -> {msg}")


log("Load data")
boundary  = gpd.read_file(BOUNDARY).to_crs(TARGET_CRS)
cand_gdf  = gpd.read_file(CANDIDATES).to_crs(TARGET_CRS)
cand_coords = np.column_stack([cand_gdf.geometry.x.values,
                                cand_gdf.geometry.y.values])

with rasterio.open(CDI_TIF) as src:
    cdi_arr   = src.read(1).astype('float64')
    nodata    = src.nodata
    transform = src.transform
cdi_arr = np.where(cdi_arr == nodata, np.nan, cdi_arr)

step = 2
dr, dc = np.where(~np.isnan(cdi_arr[::step, ::step]))
dr, dc = dr*step, dc*step
demand_x = transform.c + dc*transform.a + transform.a/2
demand_y = transform.f + dr*transform.e + transform.e/2
demand_w = cdi_arr[dr, dc]
demand_coords = np.column_stack([demand_x, demand_y])
total_demand  = demand_w.sum()
sublog(f"Candidates: {len(cand_gdf)}, demand cells: {len(demand_w)}")


def is_connected(selected, coords, max_dist=3000):
    if len(selected) <= 1: return True
    n   = len(selected)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = coords[selected[i]]
            xj, yj = coords[selected[j]]
            if np.sqrt((xi-xj)**2 + (yi-yj)**2) <= max_dist:
                adj[i].append(j); adj[j].append(i)
    visited = {0}; queue = [0]
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb); queue.append(nb)
    return len(visited) == n


def run_mclp(radius, capacity):
    demand_tree = cKDTree(demand_coords)
    covered_by  = demand_tree.query_ball_point(cand_coords, radius)
    selected    = []
    covered     = set()
    curve       = []
    prev        = 0.0

    for p in range(1, MAX_P+1):
        best_gain, best_i = -1, -1
        for i, idxs in enumerate(covered_by):
            if i in selected: continue
            if selected:
                xi, yi = cand_coords[i]
                if not any(np.sqrt((xi-cand_coords[s][0])**2 +
                                   (yi-cand_coords[s][1])**2) <= NETWORK_RADIUS
                           for s in selected):
                    continue
            new_idxs = list(set(idxs) - covered)
            if not new_idxs: continue
            new_w    = demand_w[new_idxs]
            sort_idx = np.argsort(new_w)[::-1]
            cum, gain = 0.0, 0.0
            for j in sort_idx:
                if cum + new_w[j] > capacity: break
                cum += new_w[j]; gain += new_w[j]
            if gain > best_gain:
                best_gain = gain; best_i = i

        if best_i == -1 or best_gain == 0: break

        new_idxs = list(set(covered_by[best_i]) - covered)
        new_w    = demand_w[new_idxs]
        sort_idx = np.argsort(new_w)[::-1]
        cum, done = 0.0, []
        for j in sort_idx:
            if cum + new_w[j] > capacity: break
            cum += new_w[j]; done.append(new_idxs[j])

        selected.append(best_i)
        covered.update(done)

        if not is_connected(selected, cand_coords, NETWORK_RADIUS):
            selected.pop(); covered.difference_update(done)
            continue

        rate     = demand_w[list(covered)].sum() / total_demand * 100
        marginal = rate - prev
        curve.append(rate)
        prev = rate
        sublog(f"  p={p}: {rate:.1f}% (marginal {marginal:+.1f}%)")

        if marginal < MARGINAL_THRESH and p >= 5:
            sublog(f"  Optimal p={p}, stopping")
            break

    while len(curve) < MAX_P:
        curve.append(curve[-1] if curve else 0)
    return curve[:MAX_P]


log("Run sensitivity curves")
new_curves = {}
for radius in RADII:
    label = f"sensitivity_{radius/1000:.1f}km_mid"
    sublog(f"\nRadius {radius/1000:.1f} km, C={CAPACITY_MID}")
    new_curves[label] = run_mclp(radius, CAPACITY_MID)


log("Merge with existing curves")
existing = pd.read_csv(config.CURVES_CSV)
new_df   = pd.DataFrame(new_curves)
combined = pd.concat([existing.reset_index(drop=True),
                       new_df[:len(existing)].reset_index(drop=True)], axis=1)

out_path = os.path.join(config.VERTIPORT_DIR, 'coverage_curves_complete.csv')
combined.to_csv(out_path, index=False, encoding='utf-8')
sublog(f"Saved: {out_path}")
print(combined.to_string())

log("Done")
