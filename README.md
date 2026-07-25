# feux-gironde

Source de données publique pour le suivi de l'incendie du Porge / Lège-Cap-Ferret (Gironde, saison 2026). Ce repo est lu directement (fichiers bruts GitHub, sans base de données) par une application Lovable affichant une carte interactive avec un slider temporel de la progression du feu.

## Structure

```
data/snapshots/
  manifest.json                          liste ordonnée de tous les instantanés
  {snapshot_id}/zone_brulee.geojson      zone brûlée cumulée (EPSG:4326)
  {snapshot_id}/zone_active.geojson      zone de feu actif récente, bufferisée (EPSG:4326)
```

## manifest.json

```json
{
  "snapshots": [
    {
      "fetch_timestamp": "2026-07-25T11:11:44.637537",
      "snapshot_id": "2026-07-25T11-11-44",
      "zone_brulee_geojson": "data/snapshots/2026-07-25T11-11-44/zone_brulee.geojson",
      "zone_active_geojson": "data/snapshots/2026-07-25T11-11-44/zone_active.geojson",
      "surface_brulee_ha": 18299.0,
      "nb_zones_actives": 48,
      "source": "EFFIS modis.ba.poly.season + FIRMS VIIRS NOAA20/SNPP"
    }
  ]
}
```

- `snapshots` est trié chronologiquement (ordre croissant sur `fetch_timestamp`).
- Les chemins `*_geojson` sont relatifs à la racine du repo — préfixer par l'URL raw GitHub pour les récupérer directement (ex. `https://raw.githubusercontent.com/JulienDepelchin/feux-gironde/main/<chemin>`).
- `zone_brulee.geojson` et `zone_active.geojson` sont chacun une `FeatureCollection` avec une seule `Feature` (géométrie `Polygon` ou `MultiPolygon`). Propriétés :
  - `zone_brulee` : `commune`, `firedate`, `lastupdate`, `surface_ha`, `fetch_timestamp`
  - `zone_active` : `nb_detections`, `derniere_detection`, `fetch_timestamp`

## Mise à jour

Un nouvel instantané est ajouté via `archive_snapshot.py` (dans le dépôt de travail local, pas versionné ici) — commit + push automatiques vers `main` à chaque exécution.
