"""
00_prepare_fsa.py
Phase 1 (step 0) -- Turn the raw FSA Food Hygiene Rating Scheme XML for Manchester into a
clean CSV with one WGS84 coordinate per food business.

Input : data/raw/fsa/FHRS415en-GB.xml            (FSA open data; 415 = Manchester local authority)
Output: data/processed/restaurants/manchester_fsa_complete.csv   (input of 01_restaurant_ratings.py)

Coordinate strategy (column `geocode_method` records which one was used):
  1. fsa_original       : the record already has latitude/longitude -> use as is
  2. geocoding_address  : address + postcode available -> Google Geocoding API
  3. places_text_search : no usable address -> Google Places Text Search by name
  Businesses that still cannot be located are dropped.

Filters: keep food-business types listed in FOOD_TYPES; drop records that are awaiting
inspection / publication or exempt. Low FSA ratings are kept (they only lower the demand weight later).

Requires GOOGLE_MAPS_API_KEY in the environment (only for records without coordinates).
"""

import csv
import json
import os
import time
import xml.etree.ElementTree as ET
from collections import Counter

import requests

import config
from restaurant_utils import API_KEY, is_in_manchester

INPUT_XML  = config.FSA_XML
OUTPUT_CSV = config.FSA_CSV
CACHE_FILE = os.path.join(config.RESTAURANT_DIR, 'geocode_cache.json')

FOOD_TYPES = [
    'Restaurant/Cafe/Canteen',
    'Takeaway/sandwich shop',
    'Pub/bar/nightclub',
    'Mobile caterer',
    'Other catering premises',
    'Hotel/bed & breakfast/guest house',
]
SKIP_RATINGS = ['AwaitingInspection', 'AwaitingPublication', 'Exempt']


# ------------------------------------------------------------------
# Cache (avoids repeated API calls)
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
# Geocoding helpers
# ------------------------------------------------------------------
def geocode_by_address(name, address, postcode, cache):
    """Google Geocoding API for businesses with a street address and postcode."""
    key = f"geo_{postcode}_{address[:20]}"
    if key in cache:
        r = cache[key]
        return r['lat'], r['lon'], r['method']

    params = {'address': f"{address}, {postcode}, Manchester, UK",
              'key': API_KEY, 'region': 'gb', 'components': 'country:GB'}
    try:
        data = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                            params=params, timeout=10).json()
        if data['status'] == 'OK':
            loc = data['results'][0]['geometry']['location']
            cache[key] = {'lat': loc['lat'], 'lon': loc['lng'], 'method': 'geocoding_address'}
            save_cache(cache)
            return loc['lat'], loc['lng'], 'geocoding_address'
    except Exception as e:
        print(f"  Geocoding error [{name}]: {e}")
    return None, None, None


def geocode_by_name(name, postcode, cache):
    """Google Places Text Search for businesses without any usable address."""
    key = f"places_{name[:30]}"
    if key in cache:
        r = cache[key]
        return r['lat'], r['lon'], r['method']

    query = f"{name} {postcode} Manchester" if postcode else f"{name} Manchester restaurant"
    params = {'query': query, 'key': API_KEY, 'region': 'gb'}
    try:
        data = requests.get("https://maps.googleapis.com/maps/api/place/textsearch/json",
                            params=params, timeout=10).json()
        if data['status'] == 'OK' and data['results']:
            loc = data['results'][0]['geometry']['location']
            lat, lon = loc['lat'], loc['lng']
            if is_in_manchester(lat, lon):       # loose bounding-box sanity check
                cache[key] = {'lat': lat, 'lon': lon, 'method': 'places_text_search'}
                save_cache(cache)
                return lat, lon, 'places_text_search'
            print(f"  Outside Manchester bounds [{name}]: ({lat:.4f}, {lon:.4f})")
    except Exception as e:
        print(f"  Places error [{name}]: {e}")
    return None, None, None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(config.RESTAURANT_DIR, exist_ok=True)
    cache = load_cache()
    all_ests = ET.parse(INPUT_XML).getroot().findall('.//EstablishmentDetail')
    print(f"Establishments in XML: {len(all_ests)} | cached geocodes: {len(cache)}")
    if not API_KEY:
        print("Note: GOOGLE_MAPS_API_KEY is not set; records without coordinates will be dropped.")

    results = []
    stats = {'has_coords': 0, 'geocoded_address': 0, 'geocoded_name': 0,
             'failed': 0, 'skip_type': 0, 'skip_awaiting': 0}

    for i, est in enumerate(all_ests):
        btype = est.findtext('BusinessType', '')
        if btype not in FOOD_TYPES:
            stats['skip_type'] += 1
            continue

        rating_str = est.findtext('RatingValue', '')
        if rating_str in SKIP_RATINGS:
            stats['skip_awaiting'] += 1
            continue
        try:
            fsa_rating = int(rating_str)
        except ValueError:
            fsa_rating = None

        name      = est.findtext('BusinessName', '')
        addr      = [est.findtext(k, '').strip() for k in ('AddressLine1', 'AddressLine2', 'AddressLine3')]
        address   = ', '.join(p for p in addr if p)
        postcode  = est.findtext('PostCode', '').strip()
        lat_str, lon_str = est.findtext('.//Latitude'), est.findtext('.//Longitude')

        lat = lon = method = None

        # 1. coordinates already in the FSA record
        if lat_str and lon_str and lat_str != 'None' and lon_str != 'None':
            try:
                lat, lon, method = float(lat_str), float(lon_str), 'fsa_original'
                stats['has_coords'] += 1
            except ValueError:
                lat = lon = method = None

        # 2. address + postcode -> Geocoding API
        if lat is None and address:
            lat, lon, method = geocode_by_address(name, address, postcode, cache)
            if lat:
                stats['geocoded_address'] += 1
            time.sleep(0.1)

        # 3. no address -> Places text search by name
        if lat is None:
            lat, lon, method = geocode_by_name(name, postcode, cache)
            if lat:
                stats['geocoded_name'] += 1
            else:
                stats['failed'] += 1
            time.sleep(0.1)

        if lat is None:
            continue

        results.append({
            'fhrsid': est.findtext('FHRSID', ''), 'name': name, 'business_type': btype,
            'address': address, 'postcode': postcode, 'fsa_rating': fsa_rating,
            'rating_date': est.findtext('RatingDate', ''),
            'latitude_wgs84': lat, 'longitude_wgs84': lon, 'geocode_method': method,
            'hygiene_score': est.findtext('.//Hygiene', ''),
            'structural_score': est.findtext('.//Structural', ''),
            'confidence_score': est.findtext('.//ConfidenceInManagement', ''),
        })
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(all_ests)} processed, {len(results)} kept")

    if results:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print(f"\nOriginal coordinates : {stats['has_coords']}")
    print(f"Geocoded by address  : {stats['geocoded_address']}")
    print(f"Located by name      : {stats['geocoded_name']}")
    print(f"Could not locate     : {stats['failed']}")
    print(f"Skipped (type)       : {stats['skip_type']} | skipped (awaiting/exempt): {stats['skip_awaiting']}")
    print(f"Kept                 : {len(results)}  ->  {OUTPUT_CSV}")
    print("\nBy business type:")
    for btype, n in Counter(r['business_type'] for r in results).most_common():
        print(f"  {btype}: {n}")


if __name__ == '__main__':
    main()
