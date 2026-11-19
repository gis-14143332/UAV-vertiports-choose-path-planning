"""
03_cdi_surface.py
Phase 2 -- Composite Demand Index (CDI) surface at 50 m resolution.

    CDI = 0.5 * KDE_norm + 0.5 * POP_norm

  KDE_norm : weighted kernel density of restaurants (weight = demand_weight from script 01/02,
             Silverman rule-of-thumb bandwidth), min-max normalised to [0, 1]
  POP_norm : WorldPop 2020 population density resampled to 50 m, min-max normalised
  Equal weights: a vertiport must serve both roles -- be close to restaurants (pick-up) and
             to residents (drop-off) -- so the two indicators are treated as equally important.

Input : data/processed/restaurants/manchester_restaurants_recovered.csv
        data/raw/boundary/Manchester_boundary.shp
        data/raw/worldpop/Manchester_WorldPop_100m_2020.tif
Output: data/processed/cdi/CDI_50m.tif   <- main input of 05_vertiport_siting.py
        data/processed/cdi/KDE_50m.tif, POP_50m.tif, validation_report.txt, figures/*.png

Validation checks written to validation_report.txt:
  A distribution shape, B Moran's I (spatial autocorrelation), C KDE-POP correlation,
  D hotspot location vs. city centre, E sensitivity to the KDE/POP weights.
"""

import os
import warnings

import geopandas as gpd
import libpysal
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from esda.moran import Moran
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from scipy.stats import gaussian_kde, pearsonr
from shapely.geometry import mapping

import config

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# Paths and parameters
# ------------------------------------------------------------------
CSV      = config.RESTAURANTS_CSV
BOUNDARY = config.BOUNDARY
WORLDPOP = config.WORLDPOP
OUT_DIR  = config.CDI_DIR
FIG_DIR  = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

RESOLUTION = 50
TARGET_CRS = config.TARGET_CRS
WGS84      = 'EPSG:4326'
W_KDE      = 0.5
W_POP      = 0.5
CITY_CENTRE = (383500, 398000)     # BNG, used only for the hotspot sanity check


def log(msg):
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


def sublog(msg):
    print(f"  -> {msg}")


def save_raster(array, transform, crs, path, nodata=-9999):
    out = np.where(np.isnan(array), nodata, array).astype('float32')
    with rasterio.open(path, 'w', driver='GTiff', height=array.shape[0], width=array.shape[1],
                       count=1, dtype='float32', crs=crs, transform=transform,
                       nodata=nodata, compress='lzw') as dst:
        dst.write(out, 1)
    sublog(f"Saved {os.path.basename(path)}")


def minmax_normalize(arr, mask):
    valid = arr[mask & ~np.isnan(arr)]
    vmin, vmax = valid.min(), valid.max()
    if vmax == vmin:
        return np.where(mask, 0.0, np.nan), vmin, vmax
    norm = (arr - vmin) / (vmax - vmin)
    return np.where(mask & ~np.isnan(arr), norm, np.nan), vmin, vmax


# ------------------------------------------------------------------
# Step 1: load data
# ------------------------------------------------------------------
log("Step 1: Load data")
boundary = gpd.read_file(BOUNDARY)
if boundary.crs.to_epsg() != 27700:
    boundary = boundary.to_crs(TARGET_CRS)
bounds = boundary.total_bounds

df       = pd.read_csv(CSV, encoding='utf-8')
df_valid = df[df['demand_weight'].notna()].copy()
sublog(f"Restaurants: {len(df)}, with demand_weight: {len(df_valid)}")

gdf = gpd.GeoDataFrame(df_valid,
                       geometry=gpd.points_from_xy(df_valid['final_lon'], df_valid['final_lat']),
                       crs=WGS84).to_crs(TARGET_CRS)
gdf = gpd.clip(gdf, boundary)
sublog(f"Restaurants inside boundary: {len(gdf)}")

coords_x = gdf.geometry.x.values
coords_y = gdf.geometry.y.values
weights  = gdf['demand_weight'].values

# ------------------------------------------------------------------
# Step 2: output grid (50 m cells aligned to the boundary extent)
# ------------------------------------------------------------------
log("Step 2: Build 50 m grid")
minx, miny, maxx, maxy = bounds
minx = np.floor(minx / RESOLUTION) * RESOLUTION
miny = np.floor(miny / RESOLUTION) * RESOLUTION
maxx = np.ceil(maxx / RESOLUTION) * RESOLUTION
maxy = np.ceil(maxy / RESOLUTION) * RESOLUTION

ncols = int((maxx - minx) / RESOLUTION)
nrows = int((maxy - miny) / RESOLUTION)
transform_50m = from_bounds(minx, miny, maxx, maxy, ncols, nrows)
sublog(f"Grid: {nrows} rows x {ncols} cols at {RESOLUTION} m")

xs = np.linspace(minx + RESOLUTION / 2, maxx - RESOLUTION / 2, ncols)
ys = np.linspace(maxy - RESOLUTION / 2, miny + RESOLUTION / 2, nrows)
grid_x, grid_y = np.meshgrid(xs, ys)
grid_pts = np.vstack([grid_x.ravel(), grid_y.ravel()])

boundary_mask = geometry_mask([mapping(g) for g in boundary.geometry],
                              transform=transform_50m, invert=True, out_shape=(nrows, ncols))
sublog(f"Cells inside boundary: {boundary_mask.sum()}")

# ------------------------------------------------------------------
# Step 3: weighted KDE with Silverman bandwidth
# ------------------------------------------------------------------
log("Step 3: Weighted KDE (Silverman bandwidth)")
n = len(coords_x)

# Silverman's rule: h = 0.9 * min(std, IQR/1.34) * n^(-1/5), per axis, then averaged
iqr_x = np.percentile(coords_x, 75) - np.percentile(coords_x, 25)
iqr_y = np.percentile(coords_y, 75) - np.percentile(coords_y, 25)
h_x = 0.9 * min(np.std(coords_x), iqr_x / 1.34) * (n ** (-1 / 5))
h_y = 0.9 * min(np.std(coords_y), iqr_y / 1.34) * (n ** (-1 / 5))
h_mean = (h_x + h_y) / 2
sublog(f"Silverman bandwidth: h_x={h_x:.1f} m, h_y={h_y:.1f} m, mean={h_mean:.1f} m (n={n})")

# scipy's gaussian_kde takes a bandwidth *factor* relative to the data covariance, so the data
# and the evaluation grid are standardised with the SAME mean/std, and the factor is chosen so
# that the effective bandwidth equals h_mean metres.
x_mean, x_std = coords_x.mean(), coords_x.std()
y_mean, y_std = coords_y.mean(), coords_y.std()
coords_norm = np.vstack([(coords_x - x_mean) / x_std, (coords_y - y_mean) / y_std])
grid_norm   = np.vstack([(grid_pts[0] - x_mean) / x_std, (grid_pts[1] - y_mean) / y_std])

std_pooled = np.sqrt((x_std ** 2 + y_std ** 2) / 2)
bw_factor  = h_mean / std_pooled
sublog(f"KDE bandwidth factor: {bw_factor:.4f}")

kde = gaussian_kde(coords_norm, bw_method=bw_factor, weights=weights / weights.sum())

chunk = 10000                       # evaluate in chunks to limit memory
kde_flat = np.zeros(grid_norm.shape[1])
for i in range(0, grid_norm.shape[1], chunk):
    kde_flat[i:i + chunk] = kde(grid_norm[:, i:i + chunk])

kde_masked = np.where(boundary_mask, kde_flat.reshape(nrows, ncols), np.nan)
if np.nanmax(kde_masked) == 0:
    raise ValueError("KDE is all zeros - check the coordinate system.")
save_raster(kde_masked, transform_50m, TARGET_CRS, os.path.join(OUT_DIR, 'KDE_50m.tif'))

# ------------------------------------------------------------------
# Step 4: WorldPop resampled to the 50 m grid
# ------------------------------------------------------------------
log("Step 4: Resample WorldPop to 50 m")
with rasterio.open(WORLDPOP) as src:
    pop_resampled = np.zeros((nrows, ncols), dtype='float32')
    reproject(source=rasterio.band(src, 1), destination=pop_resampled,
              src_transform=src.transform, src_crs=src.crs,
              dst_transform=transform_50m, dst_crs=TARGET_CRS,
              resampling=Resampling.bilinear)

pop_resampled = np.where(pop_resampled <= 0, np.nan, pop_resampled)
pop_masked    = np.where(boundary_mask, pop_resampled, np.nan)
save_raster(pop_masked, transform_50m, TARGET_CRS, os.path.join(OUT_DIR, 'POP_50m.tif'))

# ------------------------------------------------------------------
# Step 5-6: normalise and combine
# ------------------------------------------------------------------
log("Step 5-6: Min-max normalisation and CDI composition")
kde_norm, _, _ = minmax_normalize(kde_masked, boundary_mask)
pop_norm, _, _ = minmax_normalize(pop_masked, boundary_mask)

valid_mask = boundary_mask & ~np.isnan(kde_norm) & ~np.isnan(pop_norm)
cdi = np.where(valid_mask, W_KDE * kde_norm + W_POP * pop_norm, np.nan)
cdi_vals = cdi[valid_mask]
sublog(f"CDI: min={cdi_vals.min():.4f}, max={cdi_vals.max():.4f}, "
       f"mean={cdi_vals.mean():.4f}, std={cdi_vals.std():.4f}")
save_raster(cdi, transform_50m, TARGET_CRS, os.path.join(OUT_DIR, 'CDI_50m.tif'))

# ------------------------------------------------------------------
# Step 7: validation
# ------------------------------------------------------------------
log("Step 7: Validation")
vlines = []


def vlog(msg):
    print(f"  {msg}")
    vlines.append(msg)


# A: distribution shape
vlog("[A] Distribution shape")
skewness = float(pd.Series(cdi_vals).skew())
kurtosis = float(pd.Series(cdi_vals).kurtosis())
vlog(f"  skewness = {skewness:.3f}, excess kurtosis = {kurtosis:.3f}")
if 0.5 < skewness < 5.0:
    vlog("  PASS  right-skewed, as expected for spatially concentrated urban demand")
elif skewness >= 5.0:
    vlog("  WARN  extremely right-skewed (a few strong demand hotspots)")
else:
    vlog("  FAIL  unexpected distribution - check the inputs")
coverage = valid_mask.sum() / boundary_mask.sum() * 100
vlog(f"  valid coverage = {coverage:.1f}%  ->  {'PASS' if coverage > 95 else 'WARN'}")

# B: Moran's I on a random sample of cells (KNN, k = 8, row-standardised)
vlog("\n[B] Spatial autocorrelation (Moran's I)")
moran_I = moran_p = moran_z = None
try:
    rows_idx, cols_idx = np.where(valid_mask)
    max_sample = 8000
    if len(rows_idx) > max_sample:
        np.random.seed(42)
        s = np.random.choice(len(rows_idx), max_sample, replace=False)
        rows_s, cols_s = rows_idx[s], cols_idx[s]
    else:
        rows_s, cols_s = rows_idx, cols_idx
    xs_s = minx + cols_s * RESOLUTION + RESOLUTION / 2
    ys_s = maxy - rows_s * RESOLUTION - RESOLUTION / 2
    w = libpysal.weights.KNN.from_array(np.column_stack([xs_s, ys_s]), k=8)
    w.transform = 'r'
    moran = Moran(cdi[rows_s, cols_s], w, permutations=999)
    moran_I, moran_p, moran_z = moran.I, moran.p_norm, moran.z_norm
    vlog(f"  n = {len(rows_s)}, I = {moran_I:.4f}, z = {moran_z:.4f}, p = {moran_p:.4f}")
    if moran_I > 0.3 and moran_p < 0.05:
        vlog("  PASS  significant positive spatial autocorrelation")
    elif moran_I > 0 and moran_p < 0.05:
        vlog("  WARN  weak positive spatial autocorrelation")
    else:
        vlog("  FAIL  autocorrelation not significant")
except Exception as e:
    vlog(f"  WARN  Moran's I failed: {e}")

# C: correlation between the two components
vlog("\n[C] KDE vs population correlation (expected: moderate, 0.2-0.7)")
kde_v, pop_v = kde_norm[valid_mask], pop_norm[valid_mask]
if len(kde_v) > 10000:
    np.random.seed(42)
    sel = np.random.choice(len(kde_v), 10000, replace=False)
    r_corr, p_corr = pearsonr(kde_v[sel], pop_v[sel])
else:
    r_corr, p_corr = pearsonr(kde_v, pop_v)
vlog(f"  Pearson r = {r_corr:.4f}, p = {p_corr:.4f}")
if 0.2 <= r_corr <= 0.7 and p_corr < 0.05:
    vlog("  PASS  moderate correlation: the indicators are complementary")
elif r_corr < 0.2:
    vlog("  WARN  weak correlation: supply and demand are spatially mismatched (justifies a two-part CDI)")
elif r_corr > 0.7:
    vlog("  WARN  strong correlation: the indicators are partly redundant")

# D: hotspot location
vlog("\n[D] Hotspot plausibility (top 5% cells should lie within ~3 km of the city centre)")
threshold_95 = np.percentile(cdi_vals, 95)
hotspot_mask = valid_mask & (cdi >= threshold_95)
hs_rows, hs_cols = np.where(hotspot_mask)
if len(hs_rows) > 0:
    hs_cx = (minx + hs_cols * RESOLUTION + RESOLUTION / 2).mean()
    hs_cy = (maxy - hs_rows * RESOLUTION - RESOLUTION / 2).mean()
    dist_km = np.hypot(hs_cx - CITY_CENTRE[0], hs_cy - CITY_CENTRE[1]) / 1000
    vlog(f"  hotspot centroid (BNG) = ({hs_cx:.0f}, {hs_cy:.0f}), {dist_km:.2f} km from centre, "
         f"{hotspot_mask.sum()} cells ({hotspot_mask.sum() * RESOLUTION ** 2 / 1e6:.2f} km2)")
    vlog("  PASS" if dist_km < 3.0 else ("  WARN  slightly offset" if dist_km < 5.0 else "  FAIL  spatially implausible"))

# E: sensitivity to the weighting between KDE and POP
vlog("\n[E] Sensitivity to the KDE/POP weights")
scenarios = {
    'Equal (0.5:0.5), baseline': (W_KDE, W_POP),
    'KDE-dominant (0.7:0.3)':    (0.70, 0.30),
    'POP-dominant (0.3:0.7)':    (0.30, 0.70),
}
cdi_scenarios = {}
for name, (wk, wp) in scenarios.items():
    s = np.where(valid_mask, wk * kde_norm + wp * pop_norm, np.nan)
    cdi_scenarios[name] = s
    v = s[valid_mask]
    vlog(f"  [{name}] mean={v.mean():.4f}, std={v.std():.4f}, P90={np.percentile(v, 90):.4f}")
base_key = 'Equal (0.5:0.5), baseline'
base = cdi_scenarios[base_key][valid_mask]
r_kde, _ = pearsonr(cdi_scenarios['KDE-dominant (0.7:0.3)'][valid_mask], base)
r_pop, _ = pearsonr(cdi_scenarios['POP-dominant (0.3:0.7)'][valid_mask], base)
vlog(f"  r(KDE-dominant vs baseline) = {r_kde:.4f}")
vlog(f"  r(POP-dominant vs baseline) = {r_pop:.4f}")
vlog("  PASS  robust to the weights" if min(r_kde, r_pop) > 0.95 else
     ("  WARN  reasonably robust (r > 0.90)" if min(r_kde, r_pop) > 0.90 else
      "  FAIL  sensitive to the weights"))

vlog(f"\nSilverman bandwidth: {h_mean:.1f} m | valid cells: {len(cdi_vals):,} | "
     f"CDI mean {cdi_vals.mean():.4f}, std {cdi_vals.std():.4f}")
with open(os.path.join(OUT_DIR, 'validation_report.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(vlines))

# ------------------------------------------------------------------
# Step 8: figures
# ------------------------------------------------------------------
log("Step 8: Figures")
extent = [minx / 1000, maxx / 1000, miny / 1000, maxy / 1000]


def plot_raster(arr, title, fname, cmap, label):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.ma.masked_invalid(arr), extent=extent, cmap=cmap, origin='upper')
    plt.colorbar(im, ax=ax, label=label, shrink=0.7)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Easting (km, BNG)')
    ax.set_ylabel('Northing (km, BNG)')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()


plot_raster(kde_norm, f'Restaurant Demand KDE (normalised)\nSilverman bandwidth = {h_mean:.0f} m, n = {n}',
            'KDE_normalised.png', 'YlOrRd', 'KDE demand (0-1)')
plot_raster(pop_norm, 'Population Density (normalised)\nWorldPop 2020',
            'POP_normalised.png', 'Blues', 'Population density (0-1)')
plot_raster(cdi, f'Composite Demand Index (CDI)\nEqual weights: KDE = {W_KDE}, POP = {W_POP}',
            'CDI_final.png', 'YlOrRd', 'CDI (0-1)')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, vals, title, color in zip(axes, [kde_norm[valid_mask], pop_norm[valid_mask], cdi_vals],
                                  ['Restaurant KDE (normalised)', 'Population density (normalised)', 'CDI'],
                                  ['#e74c3c', '#3498db', '#f39c12']):
    ax.hist(vals, bins=60, color=color, alpha=0.75, edgecolor='white', linewidth=0.3)
    ax.axvline(vals.mean(), color='black', linestyle='--', linewidth=1.5, label=f'Mean = {vals.mean():.3f}')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Normalised value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle('CDI component distributions', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (name, (wk, wp)), cmap in zip(axes, scenarios.items(), ['YlOrRd', 'Reds', 'Blues']):
    im = ax.imshow(np.ma.masked_invalid(cdi_scenarios[name]), extent=extent, cmap=cmap, origin='upper')
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(f'{name}\n(KDE = {wk:.2f}, POP = {wp:.2f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Easting (km)')
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('Northing (km)')
plt.suptitle('CDI sensitivity to component weights', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sensitivity_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

log("Phase 2 complete")
print(f"Output folder: {OUT_DIR}\n"
      f"  CDI_50m.tif is the main input of 05_vertiport_siting.py\n"
      f"  Silverman bandwidth {h_mean:.1f} m | CDI mean {cdi_vals.mean():.4f} | "
      f"Moran's I {moran_I if moran_I is None else round(moran_I, 4)}")
