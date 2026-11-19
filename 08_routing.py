"""
08_routing.py
3D trajectory planning (Theta*) and multi-hop network routing (Dijkstra).

Edge-level planning:
  - Vector 3D LOS detection using STRtree against altitude-specific obstacle buildings
  - Theta* (Nash et al., 2007) with inflated heuristic epsilon = 1.2
  - Move cost = path length * mean cell soft-constraint cost

Network routing:
  - Dijkstra with mean edge risk cost as weight
  - Transfer time penalty: 120 s per intermediate vertiport

Flight parameters (Lei et al., 2024; Meituan fourth-generation UAV):
  - Vertical transition altitude: 30 m AGL at 3 m/s (10 s per phase)
  - Cruise speed: 16 m/s
  - Transfer time: 120 s
"""

import os, warnings, heapq, time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import LineString
from shapely.strtree import STRtree
from itertools import combinations
import gc

warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
import config

BLD_BASE  = config.BUILDINGS_DIR
VERTIPORT = config.VERTIPORT_SHP
OUT_DIR   = config.ROUTES_DIR
COST_MAPS = {alt: os.path.join(config.cost_dir(alt), 'cost_map_5m.tif')
             for alt in [25, 50, 80, 120]}
BLD_OBSTACLES = {alt: os.path.join(BLD_BASE, f'buildings_obstacle_{alt}m.shp')
                 for alt in [25, 50, 80, 120]}
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Parameters
# ============================================================
TARGET_CRS     = config.TARGET_CRS
RESOLUTION     = 5
ALTITUDES      = [25, 50, 80, 120]
MAX_EDGE_DIST  = 3100        # max inter-vertiport edge length (m)
TAKEOFF_ALT    = 30          # vertical transition altitude (m AGL); Lei et al. (2024)
VERTICAL_SPEED = 3.0         # vertical speed (m/s)
CRUISE_SPEED   = 16.0        # cruise speed (m/s); 70% of Meituan max
TRANSFER_TIME  = 120         # ground handling time per transfer (s)
EPSILON        = 1.2         # Theta* heuristic inflation; guarantees <= 1.2x optimal
MAX_NODES      = 2000000     # search node budget
CITY_CENTRE    = (383500, 398000)   # BNG coordinates for zone classification

# ============================================================
# Utility
# ============================================================
def log(msg):    print(f"\n{'='*60}\n{msg}\n{'='*60}")
def sublog(msg): print(f"  -> {msg}")


# ============================================================
# Step 1: Load vertiports and build valid edge list
# ============================================================
log("Step 1: Load vertiports and build edge network")
vp_gdf    = gpd.read_file(VERTIPORT).to_crs(TARGET_CRS).reset_index(drop=True)
n_vp      = len(vp_gdf)
all_pairs = list(combinations(range(n_vp), 2))

valid_edges = []
for vi, vj in all_pairs:
    xi, yi = vp_gdf.iloc[vi].geometry.x, vp_gdf.iloc[vi].geometry.y
    xj, yj = vp_gdf.iloc[vj].geometry.x, vp_gdf.iloc[vj].geometry.y
    d = np.sqrt((xj-xi)**2 + (yj-yi)**2)
    if d <= MAX_EDGE_DIST:
        valid_edges.append((vi, vj, round(d)))

sublog(f"Vertiports: {n_vp}")
sublog(f"Valid edges (<= {MAX_EDGE_DIST/1000:.1f} km): {len(valid_edges)}")


# ============================================================
# Step 2: STRtree obstacle index (per altitude, built on demand)
# ============================================================
log("Step 2: Vector building obstacle index")

def build_obstacle_index(alt):
    """Load obstacle buildings for altitude layer and build STRtree."""
    bld = gpd.read_file(BLD_OBSTACLES[alt])
    if len(bld) == 0:
        return None, None, None
    geoms   = bld.geometry.values
    heights = bld['relhmax'].values.astype('float32')
    tree    = STRtree(geoms)
    sublog(f"  {alt} m: {len(bld)} obstacle buildings, STRtree built")
    return tree, geoms, heights


# ============================================================
# Step 3: Theta* planner (closure over cost map and obstacle index)
# ============================================================
log("Step 3: Theta* path planner")

def make_planner(cost, nrows, ncols, minx, maxy, cruise_alt,
                 strtree, geoms, heights, nodata_val=-9999):
    """Return coordinate conversion and search functions for one altitude layer."""

    def w2g(x, y):
        return (int((maxy - y) / RESOLUTION), int((x - minx) / RESOLUTION))

    def g2w(r, c):
        return (minx + c*RESOLUTION + RESOLUTION/2,
                maxy - r*RESOLUTION - RESOLUTION/2)

    def passable(r, c):
        if not (0 <= r < nrows and 0 <= c < ncols): return False
        v = cost[r, c]
        return not (np.isnan(v) or v == nodata_val)

    def los_clear(r1, c1, r2, c2):
        """Vector LOS: True if the straight line does not intersect any obstacle."""
        if strtree is None: return True
        x1, y1 = g2w(r1, c1)
        x2, y2 = g2w(r2, c2)
        line = LineString([(x1, y1), (x2, y2)])
        for idx in strtree.query(line):
            if heights[idx] > cruise_alt and line.intersects(geoms[idx]):
                return False
        return True

    def search(sr, sc, gr, gcol):
        """Theta* search: any-angle path, returns list of (row, col) or None."""
        if not passable(sr, sc) or not passable(gr, gcol):
            return None
        min_c = max(float(np.nanmin(cost[~np.isnan(cost)])), 0.01)

        def h(r, c):
            return EPSILON * np.sqrt((r-gr)**2 + (c-gcol)**2) * RESOLUTION * min_c

        DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        g_score  = {(sr, sc): 0.0}
        parent   = {(sr, sc): (sr, sc)}
        open_set = [(h(sr, sc), sr, sc)]
        closed   = set()

        while open_set and len(closed) < MAX_NODES:
            _, r, c = heapq.heappop(open_set)
            node    = (r, c)
            if node in closed: continue
            closed.add(node)
            if node == (gr, gcol):
                path = []
                cur  = (gr, gcol)
                while cur != (sr, sc):
                    path.append(cur)
                    cur = parent[cur]
                path.append((sr, sc))
                gc.collect()
                return path[::-1]
            pr, pc = parent[node]
            for dr, dc in DIRS:
                nr, nc = r+dr, c+dc
                if not passable(nr, nc) or (nr, nc) in closed: continue
                if los_clear(pr, pc, nr, nc):
                    d  = np.sqrt((nr-pr)**2 + (nc-pc)**2) * RESOLUTION
                    cv = (cost[pr, pc] + cost[nr, nc]) / 2
                    ng = g_score[(pr, pc)] + d*cv
                    vp = (pr, pc)
                else:
                    d  = np.sqrt(dr**2 + dc**2) * RESOLUTION
                    cv = (cost[r, c] + cost[nr, nc]) / 2
                    ng = g_score[node] + d*cv
                    vp = node
                if (nr, nc) not in g_score or ng < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = ng
                    parent[(nr, nc)]  = vp
                    heapq.heappush(open_set, (ng + h(nr, nc), nr, nc))
        gc.collect()
        return None

    return w2g, g2w, passable, los_clear, search


# ============================================================
# Step 4: Plan paths for all valid edges at each altitude
# ============================================================
log("Step 4: Plan edge paths")
edge_results = {alt: {} for alt in ALTITUDES}

for alt in ALTITUDES:
    sublog(f"\n-- {alt} m AGL --")
    with rasterio.open(COST_MAPS[alt]) as src:
        cost   = src.read(1).astype('float32')
        nodata = src.nodata
        trans  = src.transform
        nrows, ncols = cost.shape
        minx_  = trans.c
        maxy_  = trans.f
    cost = np.where(cost == nodata, np.nan, cost)

    strtree, geoms, heights = build_obstacle_index(alt)
    w2g, g2w, passable, los_clear, search = make_planner(
        cost, nrows, ncols, minx_, maxy_, alt, strtree, geoms, heights, nodata)

    def get_vp_cell(vp_idx):
        x = vp_gdf.iloc[vp_idx].geometry.x
        y = vp_gdf.iloc[vp_idx].geometry.y
        r, c = w2g(x, y)
        if passable(r, c): return r, c
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if passable(r+dr, c+dc): return r+dr, c+dc
        return r, c

    n_ok = 0; n_blocked = 0; t0 = time.time()

    for i, (vi, vj, edge_dist) in enumerate(valid_edges):
        r1, c1 = get_vp_cell(vi)
        r2, c2 = get_vp_cell(vj)
        xi, yi = vp_gdf.iloc[vi].geometry.x, vp_gdf.iloc[vi].geometry.y
        xj, yj = vp_gdf.iloc[vj].geometry.x, vp_gdf.iloc[vj].geometry.y

        direct_los = los_clear(r1, c1, r2, c2)
        path       = search(r1, c1, r2, c2)

        if path:
            coords  = [g2w(r, c) for r, c in path]
            xs, ys  = zip(*coords)
            segs    = [np.sqrt((xs[k+1]-xs[k])**2 + (ys[k+1]-ys[k])**2)
                       for k in range(len(coords)-1)]
            length  = sum(segs)
            c_vals  = [float(cost[r, c]) for r, c in path if not np.isnan(cost[r, c])]
            mean_c  = float(np.mean(c_vals)) if c_vals else 0.3
            edge_results[alt][(vi, vj)] = {
                'coords'     : list(zip(xs, ys)),
                'length_m'   : round(length, 1),
                'mean_cost'  : round(mean_c, 4),
                'n_waypoints': len(path),
                'direct_los' : direct_los,
                'geometry'   : LineString(list(zip(xs, ys))),
            }
            n_ok += 1
            if not direct_los: n_blocked += 1
        else:
            # Theta* failed: sample straight-line cost
            n_s   = max(int(edge_dist / RESOLUTION), 2)
            s_vals = []
            for k in range(n_s):
                t   = k / (n_s - 1)
                px  = xi + t*(xj-xi); py = yi + t*(yj-yi)
                r_, c_ = w2g(px, py)
                if 0 <= r_ < nrows and 0 <= c_ < ncols:
                    v = float(cost[r_, c_])
                    if not np.isnan(v): s_vals.append(v)
            mean_c = float(np.mean(s_vals)) if s_vals else 0.3
            edge_results[alt][(vi, vj)] = {
                'coords'     : [(xi, yi), (xj, yj)],
                'length_m'   : float(edge_dist),
                'mean_cost'  : round(mean_c, 4),
                'n_waypoints': 2,
                'direct_los' : direct_los,
                'geometry'   : LineString([(xi, yi), (xj, yj)]),
            }

        if (i+1) % 5 == 0:
            elapsed = time.time() - t0
            remain  = (len(valid_edges) - i - 1) / (i + 1) * elapsed / 60
            sublog(f"  {i+1}/{len(valid_edges)} ok:{n_ok} blocked:{n_blocked} "
                   f"est_remain:{remain:.1f} min")

    gc.collect()

    # Save edge paths as GeoPackage
    attrs = []
    geoms_out = []
    for vi, vj, d in valid_edges:
        if (vi, vj) not in edge_results[alt]: continue
        r = edge_results[alt][(vi, vj)]
        attrs.append({'vi': vi+1, 'vj': vj+1, 'edge_dist_m': d,
                      'path_len_m': r['length_m'], 'mean_cost': r['mean_cost'],
                      'altitude_m': alt, 'n_waypoints': r['n_waypoints'],
                      'direct_los': r['direct_los']})
        geoms_out.append(r['geometry'])

    gdf_out = gpd.GeoDataFrame(attrs, geometry=geoms_out, crs=TARGET_CRS)
    gdf_out.to_file(os.path.join(OUT_DIR, f'edge_paths_{alt}m.gpkg'), driver='GPKG')
    sublog(f"  Saved: edge_paths_{alt}m.gpkg")
    sublog(f"  LOS blocked: {n_blocked}/{len(valid_edges)}")
    sublog(f"  Mean path length: "
           f"{np.mean([r['length_m'] for r in edge_results[alt].values()]):.0f} m")


# ============================================================
# Step 5: Dijkstra multi-hop routing
# ============================================================
log("Step 5: Dijkstra network routing")

def dijkstra(start, end, edge_costs, n_nodes):
    dist = {i: float('inf') for i in range(n_nodes)}
    prev = {i: None for i in range(n_nodes)}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == end:
            route = []; cur = end
            while cur is not None:
                route.append(cur); cur = prev[cur]
            return route[::-1], dist[end]
        for (vi, vj), c in edge_costs.items():
            v = vj if vi == u else (vi if vj == u else None)
            if v is None: continue
            if dist[u] + c < dist[v]:
                dist[v] = dist[u] + c; prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    return None, float('inf')


all_routes = {alt: {} for alt in ALTITUDES}

for alt in ALTITUDES:
    edge_costs = {(vi, vj): edge_results[alt][(vi, vj)]['mean_cost']
                  for vi, vj, _ in valid_edges
                  if (vi, vj) in edge_results[alt]}
    n_ok = 0
    for vi, vj in all_pairs:
        route, total_cost = dijkstra(vi, vj, edge_costs, n_vp)
        if route is None: continue
        hops         = len(route) - 1
        total_length = sum(
            edge_results[alt].get(
                (min(route[k], route[k+1]), max(route[k], route[k+1])), {}
            ).get('length_m', 0)
            for k in range(hops))
        cruise_t   = total_length / CRUISE_SPEED
        vertical_t = TAKEOFF_ALT / VERTICAL_SPEED * 2
        transfer_t = (hops - 1) * TRANSFER_TIME
        total_t    = cruise_t + vertical_t + transfer_t
        mid_x = (vp_gdf.iloc[vi].geometry.x + vp_gdf.iloc[vj].geometry.x) / 2
        mid_y = (vp_gdf.iloc[vi].geometry.y + vp_gdf.iloc[vj].geometry.y) / 2
        dist_c = np.sqrt((mid_x - CITY_CENTRE[0])**2 +
                          (mid_y - CITY_CENTRE[1])**2) / 1000
        all_routes[alt][(vi, vj)] = {
            'route'          : route,
            'hops'           : hops,
            'total_length_m' : round(total_length, 1),
            'total_cost'     : round(total_cost, 4),
            'total_time_s'   : round(total_t, 1),
            'total_time_min' : round(total_t / 60, 2),
            'dist_centre_km' : round(dist_c, 2),
        }
        n_ok += 1
    sublog(f"{alt} m: {n_ok}/{len(all_pairs)} routes successful")


# ============================================================
# Step 6: Summary report
# ============================================================
log("Step 6: Summary")
for alt in ALTITUDES:
    r = edge_results[alt]
    if not r: continue
    los_blocked = sum(1 for v in r.values() if not v['direct_los'])
    costs       = [v['mean_cost'] for v in r.values()]
    rt          = all_routes[alt]
    times       = [v['total_time_min'] for v in rt.values()]
    hops        = [v['hops'] for v in rt.values()]
    print(f"  {alt} m | edges:{len(r)} los_blocked:{los_blocked} "
          f"mean_cost:{np.mean(costs):.4f} "
          f"mean_time:{np.mean(times):.2f} min "
          f"mean_hops:{np.mean(hops):.1f}")


# ============================================================
# Step 7: Save routing CSV
# ============================================================
log("Step 7: Save all_routes_new.csv")
rows = []
for alt in ALTITUDES:
    for (vi, vj), r in all_routes[alt].items():
        rows.append({
            'altitude_m'     : alt,
            'vi'             : vi+1, 'vj': vj+1,
            'route'          : str([v+1 for v in r['route']]),
            'hops'           : r['hops'],
            'total_cost'     : r['total_cost'],
            'total_length_m' : r['total_length_m'],
            'total_time_s'   : r['total_time_s'],
            'total_time_min' : r['total_time_min'],
            'dist_centre_km' : r['dist_centre_km'],
        })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, 'all_routes_new.csv'), index=False, encoding='utf-8')
sublog(f"Saved: all_routes_new.csv ({len(df)} rows)")

log("Phase 5 complete")
