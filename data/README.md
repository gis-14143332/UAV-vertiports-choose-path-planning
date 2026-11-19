# Data

No data are stored in this repository (files are large and several are licensed).
Download each dataset yourself and place it as shown. Folder names are fixed in `config.py`;
set the environment variable `UAV_DATA_DIR` if you want the `data/` folder somewhere else.

```
data/
  raw/                                   <- you provide these
    boundary/Manchester_boundary.shp
    fsa/FHRS415en-GB.xml
    worldpop/Manchester_WorldPop_100m_2020.tif
    os_building_heights/
      download_1/{sd,sj,sk}/*.gdb        <- OS Building Heights tiles (several downloads)
      download_2/...
      download_3/...
    osm/gis_osm_roads_free_1.shp  (+ .dbf .shx .prj)
    osm/gis_osm_railways_free_1.shp (+ sidecars)
    caa/EG_UAS_FR_DS_AREA1_FULL*.kmz
    icao/EG_OBS_DS_AREA1_FULL*.xls
    wind/*.nc
  processed/                             <- written by the scripts (one folder per stage)
```

| Dataset | Used for | Source | Licence / access |
|---|---|---|---|
| Food hygiene ratings (FHRS XML) | restaurant list, addresses, FSA rating (script 00) | Food Standards Agency, Food Hygiene Rating Scheme open data, Manchester local authority file `FHRS415en-GB.xml` (food-hygiene-ratings download page on ratings.food.gov.uk). Records without coordinates are geocoded in script 00 from the address (`geocode_method` = `geocoding_address`) or by name (`places_text_search`). | Open Government Licence |
| Google Geocoding and Places (Text Search) | geocoding of records without coordinates (script 00); Google rating, review count, accurate coordinates (scripts 01-02) | Google Maps Platform | Requires an API key and billing; results are not redistributed here |
| WorldPop 100 m population, 2020 | population component of the CDI (script 03) | WorldPop, UK 100 m constrained/unconstrained 2020 raster, clipped to Manchester | CC BY 4.0 |
| OS Building Heights (Attribute) | 237,782 building footprints with `relhmax` height (scripts 04-07) | Ordnance Survey, via EDINA Digimap | Academic licence: cannot be redistributed |
| OpenStreetMap roads and railways | road-distance criterion; light-rail EMI layer (scripts 05, 07) | Geofabrik Greater Manchester extract (`gis_osm_roads_free_1`, `gis_osm_railways_free_1`) | ODbL |
| UK UAS restricted areas | hard no-fly constraint (script 05) | UK AIP dataset `EG_UAS_FR_DS_AREA1_FULL` (CAA / NATS AIS) | UK AIS terms |
| ICAO Area 1 obstacle data | tall man-made obstacles, hard constraint (scripts 05, 07) | UK AIP dataset `EG_OBS_DS_AREA1_FULL` | UK AIS terms |
| HadUK-Grid wind speed (NetCDF) | wind-risk layer, 7-year mean (script 07) | Met Office HadUK-Grid via CEDA | OGL |
| Manchester City boundary | study area mask (all scripts) | Local-authority boundary for Manchester City (e.g. ONS Open Geography Portal), EPSG:27700 | OGL |

## Notes

- All spatial work is done in the British National Grid (EPSG:27700).
- Script 05 (siting) reads the tiles in `download_1` only, as in the original analysis;
  scripts 04 and 07-09 use the merged building layer from all downloads.
- Do not commit the Google API key. Export it as an environment variable instead:
  `export GOOGLE_MAPS_API_KEY="..."`.
