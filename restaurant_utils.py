"""
Shared helpers for the two restaurant-matching scripts
(01_restaurant_ratings.py and 02_restaurant_recovery.py).
"""

import math
import os
from math import asin, cos, radians, sin, sqrt

import requests
from fuzzywuzzy import fuzz

# Google Places API key: read from the environment, never hard-code it.
API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', '')

MANCHESTER_BOUNDS = {              # loose bounding box around Manchester City
    'lat_min': 53.30, 'lat_max': 53.60,
    'lon_min': -2.40, 'lon_max': -2.10,
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * R * asin(sqrt(a))


def is_in_manchester(lat, lon):
    return (MANCHESTER_BOUNDS['lat_min'] <= lat <= MANCHESTER_BOUNDS['lat_max'] and
            MANCHESTER_BOUNDS['lon_min'] <= lon <= MANCHESTER_BOUNDS['lon_max'])


def name_similarity(name1, name2):
    """Token-sort ratio on lower-cased names, 0-100."""
    return fuzz.token_sort_ratio(str(name1).lower().strip(), str(name2).lower().strip())


def text_search(query, lat=None, lon=None, radius=None):
    """Google Places Text Search; returns the list of results ([] on any failure)."""
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {'query': query, 'key': API_KEY, 'region': 'gb'}
    if lat is not None and lon is not None:
        params['location'] = f"{lat},{lon}"
    if radius is not None:
        params['radius'] = radius
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get('results', []) if data['status'] == 'OK' else []
    except Exception:
        return []


def calculate_demand_weight(google_rating, review_count, fsa_rating):
    """
    demand_weight = (0.667 * google_rating/5 + 0.333 * fsa_rating/5) * ln(1 + review_count)

    0.667 / 0.333 : AHP weights from a 2x2 pairwise matrix (Saaty, 1980)
    ln(1 + n)     : log transform of the skewed review-count distribution
    Returns None when there is no Google rating (the restaurant is excluded from KDE).
    FSA rating 0 is kept: it just lowers the weight.
    """
    if google_rating is None:
        return None
    quality = (0.667 * google_rating / 5.0 +
               0.333 * (fsa_rating / 5.0 if fsa_rating is not None else 0))
    return round(quality * math.log1p(review_count), 4)
