"""
01_restaurant_ratings.py
Phase 1a -- Match FSA-registered food businesses to Google Maps places and compute
a demand weight for each restaurant.

Input : data/processed/restaurants/manchester_fsa_complete.csv   (from 00_prepare_fsa.py)
Output: data/processed/restaurants/manchester_restaurants_final.csv

Requires a Google Places API key in the environment variable GOOGLE_MAPS_API_KEY.
Never hard-code the key in this file or commit it to GitHub.

Matching rule (a Google result is accepted only if BOTH hold):
  - name similarity   >= 60 (fuzzywuzzy token-sort ratio, 0-100)
  - spatial distance  <= 300 m from the FSA coordinate (FSA coordinates are only
    accurate to roughly 200 m, so the usual 100 m tolerance is relaxed)
This follows the two-criteria approach used in POI conflation (see Sun et al., 2023).

Search strategy (stops at the first step that finds a match, to save API calls):
  1. cleaned name + FSA coordinate, 500 m
  2. original name + FSA coordinate, 500 m (only if cleaning changed the name)
  3. cleaned name + FSA coordinate, 1000 m (distance threshold 500 m)
  4. cleaned name + postcode (no coordinate bias; must fall inside Manchester bounds)

Final coordinate: Google coordinate (about +/-10 m) when matched, else the FSA one.
Restaurants with no Google record keep demand_weight = NaN and are excluded from KDE.

Demand weight (documented in the dissertation, Section 3.2):
  demand_weight = (0.667 * google_rating/5 + 0.333 * fsa_rating/5) * ln(1 + review_count)
  - 0.667 / 0.333 : AHP weights from a 2x2 pairwise matrix (Saaty, 1980)
  - ln(1 + n)     : log transform to compress the skewed review-count distribution
"""

import json
import os
import re
import time

import pandas as pd

import config
from restaurant_utils import (API_KEY, calculate_demand_weight, haversine_distance,
                              is_in_manchester, name_similarity, text_search)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
INPUT_CSV  = config.FSA_CSV
OUTPUT_CSV = config.RATINGS_CSV
CACHE_FILE = os.path.join(config.RESTAURANT_DIR, 'ratings_cache_final.json')

NAME_SIMILARITY_THRESHOLD = 0.6    # 60 / 100
DISTANCE_THRESHOLD_M      = 300


# ------------------------------------------------------------------
# Cache (API results are stored so the script can be re-run cheaply)
# ------------------------------------------------------------------
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def clean_name(name):
    """Turn an FSA legal/registered name into a likely trading name."""
    clean = str(name).strip()

    # Rule 1: drop "also T/as ..." (trading-as) suffixes
    for pattern in ['also T/as', 'Also T/as', 'also t/as', ' T/as ', ' t/as ']:
        if pattern in clean:
            clean = clean.split(pattern)[0].strip()

    # Rule 2: drop bracketed text
    clean = re.sub(r'\([^)]*\)', '', clean).strip()
    clean = re.sub(r'\[[^\]]*\]', '', clean).strip()

    # Rule 3: drop company suffixes
    clean = re.sub(r'\s+(Ltd|Limited|Ltd\.|LTD|LLC|PLC|plc|& Co|and Co)\.?$',
                   '', clean, flags=re.IGNORECASE).strip()

    # Rule 4: for long names, keep the part before " - "
    if len(clean) > 30 and ' - ' in clean:
        clean = clean.split(' - ')[0].strip()

    return clean if clean else str(name).strip()


def find_best_match(results, query_name, fsa_lat, fsa_lon,
                    distance_threshold=DISTANCE_THRESHOLD_M,
                    similarity_threshold=NAME_SIMILARITY_THRESHOLD):
    """Best of the top-5 results that satisfies BOTH the name and distance criteria."""
    best, best_score = None, -1
    for r in results[:5]:
        loc = r.get('geometry', {}).get('location', {})
        r_lat, r_lon = loc.get('lat'), loc.get('lng')
        if r_lat is None or r_lon is None or not is_in_manchester(r_lat, r_lon):
            continue

        sim  = name_similarity(query_name, r.get('name', ''))
        dist = haversine_distance(fsa_lat, fsa_lon, r_lat, r_lon)

        if sim >= similarity_threshold * 100 and dist <= distance_threshold:
            score = sim - dist / 10        # higher similarity and shorter distance win
            if score > best_score:
                best_score, best = score, r
    return best


def get_google_info(name, fsa_lat, fsa_lon, postcode, cache):
    """Four-step search; returns (rating, n_reviews, place_id, lat, lon, method)."""
    cache_key = f"final_{fsa_lat:.4f}_{fsa_lon:.4f}_{name[:25]}"
    if cache_key in cache:
        r = cache[cache_key]
        return (r['google_rating'], r['review_count'], r['place_id'],
                r['google_lat'], r['google_lon'], r['method'])

    cleaned = clean_name(name)
    result, method = None, None

    # Step 1: cleaned name, 500 m
    res = text_search(cleaned, lat=fsa_lat, lon=fsa_lon, radius=500)
    time.sleep(0.1)
    result = find_best_match(res, cleaned, fsa_lat, fsa_lon)
    if result:
        method = 'step1_clean+coords500m'

    # Step 2: original name, 500 m (only if cleaning changed it)
    if result is None and cleaned != name:
        res = text_search(name, lat=fsa_lat, lon=fsa_lon, radius=500)
        time.sleep(0.1)
        result = find_best_match(res, name, fsa_lat, fsa_lon)
        if result:
            method = 'step2_original+coords500m'

    # Step 3: cleaned name, 1000 m search area, 500 m distance threshold
    if result is None:
        res = text_search(cleaned, lat=fsa_lat, lon=fsa_lon, radius=1000)
        time.sleep(0.1)
        result = find_best_match(res, cleaned, fsa_lat, fsa_lon, distance_threshold=500)
        if result:
            method = 'step3_clean+coords1000m'

    # Step 4: cleaned name + postcode, no distance check (Manchester bounds + name only)
    if result is None and postcode and str(postcode) != 'nan':
        res = text_search(f"{cleaned} {postcode}")
        time.sleep(0.1)
        for r in res[:3]:
            loc = r.get('geometry', {}).get('location', {})
            r_lat, r_lon = loc.get('lat'), loc.get('lng')
            if r_lat and r_lon and is_in_manchester(r_lat, r_lon):
                if name_similarity(cleaned, r.get('name', '')) >= NAME_SIMILARITY_THRESHOLD * 100:
                    result, method = r, 'step4_clean+postcode'
                    break

    if result:
        loc = result['geometry']['location']
        out = (result.get('rating'), result.get('user_ratings_total', 0),
               result.get('place_id', ''), loc['lat'], loc['lng'])
    else:
        out = (None, 0, '', None, None)

    cache[cache_key] = {'google_rating': out[0], 'review_count': out[1], 'place_id': out[2],
                        'google_lat': out[3], 'google_lon': out[4], 'method': method}
    save_cache(cache)
    return (*out, method)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    if not API_KEY:
        raise SystemExit("Set the GOOGLE_MAPS_API_KEY environment variable first.")
    os.makedirs(config.RESTAURANT_DIR, exist_ok=True)

    cache = load_cache()
    df = pd.read_csv(INPUT_CSV, encoding='utf-8')
    total = len(df)
    print(f"Restaurants: {total} | cached lookups: {len(cache)}")

    df['cleaned_name']  = df['name'].apply(clean_name)
    df['google_rating'] = None
    df['review_count']  = 0
    df['place_id']      = ''
    df['search_method'] = ''
    df['google_lat']    = None
    df['google_lon']    = None
    df['final_lat']     = None   # coordinates used for KDE
    df['final_lon']     = None
    df['demand_weight'] = None

    stats = {'found': 0, 'not_found': 0, 'step1': 0, 'step2': 0, 'step3': 0, 'step4': 0}

    for i, row in df.iterrows():
        name     = str(row['name'])
        fsa_lat  = float(row['latitude_wgs84'])
        fsa_lon  = float(row['longitude_wgs84'])
        postcode = str(row.get('postcode', ''))

        (rating, n_reviews, place_id,
         g_lat, g_lon, method) = get_google_info(name, fsa_lat, fsa_lon, postcode, cache)

        df.at[i, 'google_rating'] = rating
        df.at[i, 'review_count']  = n_reviews
        df.at[i, 'place_id']      = place_id
        df.at[i, 'search_method'] = method or ''

        if g_lat is not None:
            df.at[i, 'google_lat'], df.at[i, 'google_lon'] = g_lat, g_lon
            df.at[i, 'final_lat'],  df.at[i, 'final_lon']  = g_lat, g_lon
        else:
            df.at[i, 'final_lat'],  df.at[i, 'final_lon']  = fsa_lat, fsa_lon

        df.at[i, 'demand_weight'] = calculate_demand_weight(rating, n_reviews, row['fsa_rating'])

        if rating is not None:
            stats['found'] += 1
            for k in ('step1', 'step2', 'step3', 'step4'):
                if k in (method or ''):
                    stats[k] += 1
                    break
        else:
            stats['not_found'] += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{total} | matched {stats['found']} "
                  f"({stats['found'] / (i + 1):.1%}) | not found {stats['not_found']}")

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    print(f"\nMatched   : {stats['found']} ({stats['found'] / total:.1%})")
    print(f"Unmatched : {stats['not_found']} ({stats['not_found'] / total:.1%})")
    print(f"By step   : {stats['step1']} / {stats['step2']} / {stats['step3']} / {stats['step4']}")
    print(f"Saved     : {OUTPUT_CSV}")
    print("Columns used downstream: final_lat/final_lon (KDE location), demand_weight (KDE weight)")


if __name__ == '__main__':
    main()
