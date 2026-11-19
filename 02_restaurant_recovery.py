"""
02_restaurant_recovery.py
Phase 1b -- Second-pass matching for FSA restaurants that script 01 could not find on
Google Maps. Two targeted strategies are applied to the unmatched records:

  A. Chain brands (KFC, Nando's, Subway ...): several branches share one name, so the
     name-plus-distance test of script 01 fails. Search "brand + street keywords" and
     accept a result whose name CONTAINS the brand and lies within 500-600 m.

  B. Regular restaurants whose FSA legal name differs from the Google trading name:
       B1  "T/as" alias search (e.g. "Alberts Shed ... T/as Dukes 92" -> "Dukes 92")
       B2  cleaned name + street keywords, relaxed similarity (>= 45) but distance < 400 m
       B3  records geocoded from the address only: relax distance to 800 m, similarity >= 60
       B4  cleaned name + postcode, similarity >= 45 and distance < 500 m
     Restaurants that are closed or not listed on Google remain unmatched and are excluded.

Input : data/processed/restaurants/manchester_restaurants_final.csv  (from script 01)
Output: data/processed/restaurants/manchester_restaurants_recovered.csv
        (drop-in replacement of the input; this is the file used by the CDI model)

Requires GOOGLE_MAPS_API_KEY in the environment.
"""

import json
import os
import re
import time

import pandas as pd

import config
from restaurant_utils import (API_KEY, calculate_demand_weight, haversine_distance,
                              is_in_manchester, name_similarity, text_search)

INPUT_CSV  = config.RATINGS_CSV
OUTPUT_CSV = config.RESTAURANTS_CSV
CACHE_FILE = os.path.join(config.RESTAURANT_DIR, 'ratings_cache_recovery.json')
PAUSE      = 0.12   # seconds between API calls

CHAIN_KEYWORDS = [
    'KFC', "Nando's", "Nandos", "Domino's Pizza", "Dominos Pizza",
    'Pizza Express', 'Barburrito', 'Starbucks', 'Wagamama',
    'Greggs', 'Subway', 'Costa', "McDonald's", 'Five Guys',
    'Pret', 'Burger King', 'Archies',
]


# ------------------------------------------------------------------
# Cache and name utilities
# ------------------------------------------------------------------
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def extract_address_keywords(address):
    """First address line without generic words (Manchester, Road, Street ...), max 3 words."""
    if not address or str(address) == 'nan':
        return ''
    skip = {'manchester', 'uk', 'england', 'greater', 'road', 'street',
            'avenue', 'lane', 'drive', 'close', 'way', 'place'}
    first_line = re.split(r'[,\n]', str(address))[0].strip()
    kept = [w for w in first_line.split()[:4] if w.lower() not in skip][:3]
    return ' '.join(kept)


def extract_alias(name):
    """Trading name after 'T/as' (trading as), or None."""
    patterns = ['also T/as', 'Also T/as', 'also t/as', ' T/as ', ' t/as ',
                'also T/a ', 'also t/a ', 'T/as ', 't/as ']
    name_lower = name.lower()
    for p in patterns:
        if p.lower() in name_lower:
            idx = name_lower.index(p.lower())
            alias = name[idx + len(p):].strip()
            alias = re.sub(r',.*$', '', alias).strip()                     # first alias only
            alias = re.sub(r'\s+(Ltd|Limited|LTD|LLC|PLC)\.?$', '',
                           alias, flags=re.IGNORECASE).strip()
            return alias if len(alias) > 2 else None
    return None


def chain_brand(name):
    """Return the chain keyword contained in the name, or None."""
    name_lower = name.lower()
    for chain in CHAIN_KEYWORDS:
        if chain.lower() in name_lower:
            return chain
    return None


# ------------------------------------------------------------------
# Search helpers
# ------------------------------------------------------------------
def pick(results, fsa_lat, fsa_lon, max_dist, min_sim=None, query_name=None, brand=None):
    """First of the top-5 results inside Manchester that satisfies the distance limit and
    either the brand-in-name test (brand given) or the similarity test (min_sim given)."""
    for r in results[:5]:
        loc = r['geometry']['location']
        r_lat, r_lon = loc['lat'], loc['lng']
        if not is_in_manchester(r_lat, r_lon):
            continue
        if haversine_distance(fsa_lat, fsa_lon, r_lat, r_lon) >= max_dist:
            continue
        if brand is not None:
            if brand.lower() in r.get('name', '').lower():
                return r
        elif name_similarity(query_name, r.get('name', '')) >= min_sim:
            return r
    return None


def unpack(result, method):
    """Convert a Google result (or None) into the tuple stored in the cache / CSV."""
    if result is None:
        return None, 0, None, None, None, method
    loc = result['geometry']['location']
    return (result.get('rating'), result.get('user_ratings_total', 0),
            result.get('place_id', ''), loc['lat'], loc['lng'], method)


def cached_search(cache, key, finder):
    """Run `finder()` -> (result, method) once per key and cache the unpacked tuple."""
    if key in cache:
        r = cache[key]
        return (r['google_rating'], r['review_count'], r['place_id'],
                r['google_lat'], r['google_lon'], r['method'])
    result, method = finder()
    out = unpack(result, method)
    cache[key] = {'google_rating': out[0], 'review_count': out[1], 'place_id': out[2],
                  'google_lat': out[3], 'google_lon': out[4], 'method': out[5]}
    save_cache(cache)
    return out


# ------------------------------------------------------------------
# Strategy A: chain brands
# ------------------------------------------------------------------
def search_chain(name, address, fsa_lat, fsa_lon, postcode, brand, cache):
    def finder():
        addr_kw = extract_address_keywords(address)
        # A1: brand + street keywords (most precise); name must contain the brand, < 500 m
        if addr_kw:
            res = text_search(f"{brand} {addr_kw} Manchester",
                              lat=fsa_lat, lon=fsa_lon, radius=800)
            time.sleep(PAUSE)
            hit = pick(res, fsa_lat, fsa_lon, 500, brand=brand)
            if hit:
                return hit, 'recovery_chain_addr'
        # A2: brand + postcode, < 600 m
        if postcode and postcode != 'nan':
            res = text_search(f"{brand} {postcode}")
            time.sleep(PAUSE)
            hit = pick(res, fsa_lat, fsa_lon, 600, brand=brand)
            if hit:
                return hit, 'recovery_chain_postcode'
        return None, None

    return cached_search(cache, f"recovery_chain_{fsa_lat:.4f}_{fsa_lon:.4f}_{name[:20]}", finder)


# ------------------------------------------------------------------
# Strategy B: regular restaurants with name mismatch
# ------------------------------------------------------------------
def search_regular(name, cleaned_name, address, fsa_lat, fsa_lon,
                   postcode, geocode_method, cache):
    def finder():
        alias   = extract_alias(name)
        addr_kw = extract_address_keywords(address)
        has_pc  = bool(postcode) and postcode != 'nan'

        # B1: T/as alias (highest priority)
        if alias and len(alias) > 3:
            res = text_search(alias, lat=fsa_lat, lon=fsa_lon, radius=600)
            time.sleep(PAUSE)
            hit = pick(res, fsa_lat, fsa_lon, 500, min_sim=55, query_name=alias)
            if hit:
                return hit, 'recovery_alias_coords'
            if has_pc:
                res = text_search(f"{alias} {postcode}")
                time.sleep(PAUSE)
                hit = pick(res, fsa_lat, fsa_lon, 600, min_sim=55, query_name=alias)
                if hit:
                    return hit, 'recovery_alias_postcode'

        # B2: name + street keywords, similarity >= 45 but distance < 400 m (avoids false matches)
        if addr_kw:
            res = text_search(f"{cleaned_name} {addr_kw} Manchester",
                              lat=fsa_lat, lon=fsa_lon, radius=600)
            time.sleep(PAUSE)
            hit = pick(res, fsa_lat, fsa_lon, 400, min_sim=45, query_name=cleaned_name)
            if hit:
                return hit, 'recovery_addr_lowthresh'

        # B3: coordinates geocoded from the address are less reliable -> distance up to 800 m,
        #     compensated by a stricter similarity (>= 60)
        if geocode_method == 'geocoding_address' and has_pc:
            res = text_search(f"{cleaned_name} {postcode}")
            time.sleep(PAUSE)
            hit = pick(res, fsa_lat, fsa_lon, 800, min_sim=60, query_name=cleaned_name)
            if hit:
                return hit, 'recovery_geocoded_relaxed'

        # B4: last attempt, name + postcode, similarity >= 45 and distance < 500 m
        if has_pc:
            res = text_search(f"{cleaned_name} {postcode}")
            time.sleep(PAUSE)
            hit = pick(res, fsa_lat, fsa_lon, 500, min_sim=45, query_name=cleaned_name)
            if hit:
                return hit, 'recovery_postcode_lowthresh'

        return None, None

    return cached_search(cache, f"recovery_reg_{fsa_lat:.4f}_{fsa_lon:.4f}_{name[:20]}", finder)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def write_match(df, idx, google_rating, review_count, place_id, g_lat, g_lon, method, fsa):
    df.at[idx, 'google_rating'] = google_rating
    df.at[idx, 'review_count']  = review_count
    df.at[idx, 'place_id']      = place_id
    df.at[idx, 'search_method'] = method
    df.at[idx, 'google_lat']    = g_lat
    df.at[idx, 'google_lon']    = g_lon
    df.at[idx, 'final_lat']     = g_lat      # Google coordinate replaces the FSA one
    df.at[idx, 'final_lon']     = g_lon
    df.at[idx, 'demand_weight'] = calculate_demand_weight(google_rating, review_count, fsa)


def main():
    if not API_KEY:
        raise SystemExit("Set the GOOGLE_MAPS_API_KEY environment variable first.")

    cache = load_cache()
    df = pd.read_csv(INPUT_CSV, encoding='utf-8')

    found_mask    = df['google_rating'].notna()
    already_found = df[found_mask].copy()
    not_found     = df[~found_mask].copy()
    print(f"Total {len(df)} | already matched {len(already_found)} | to recover {len(not_found)}")

    is_chain  = not_found['name'].apply(lambda n: chain_brand(str(n)) is not None)
    cat_chain, cat_regular = not_found[is_chain].copy(), not_found[~is_chain].copy()
    print(f"Strategy A (chains): {len(cat_chain)} | Strategy B (regular): {len(cat_regular)}")

    n_a = n_b = 0
    for idx, row in cat_chain.iterrows():
        name = str(row['name'])
        out = search_chain(name, str(row.get('address', '')),
                           float(row['latitude_wgs84']), float(row['longitude_wgs84']),
                           str(row.get('postcode', '')), chain_brand(name), cache)
        if out[0] is not None:
            write_match(cat_chain, idx, *out, row['fsa_rating'])
            n_a += 1
    print(f"Strategy A recovered {n_a}/{len(cat_chain)}")

    for i, (idx, row) in enumerate(cat_regular.iterrows(), 1):
        name = str(row['name'])
        out = search_regular(name, str(row.get('cleaned_name', name)),
                             str(row.get('address', '')),
                             float(row['latitude_wgs84']), float(row['longitude_wgs84']),
                             str(row.get('postcode', '')),
                             str(row.get('geocode_method', '')), cache)
        if out[0] is not None:
            write_match(cat_regular, idx, *out, row['fsa_rating'])
            n_b += 1
        if i % 100 == 0:
            print(f"  B: {i}/{len(cat_regular)} processed, {n_b} recovered")
    print(f"Strategy B recovered {n_b}/{len(cat_regular)}")

    recovered = pd.concat([already_found, cat_chain, cat_regular]).sort_index()
    recovered.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    total, final_found = len(recovered), recovered['google_rating'].notna().sum()
    print(f"\nMatched before: {len(already_found)} ({len(already_found) / total:.1%})")
    print(f"Matched after : {final_found} ({final_found / total:.1%}) | still unmatched {total - final_found}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
