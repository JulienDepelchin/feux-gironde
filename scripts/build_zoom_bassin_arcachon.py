"""Carte zoomee "Incendies en Gironde" - secteur nord du Bassin d'Arcachon
(Lege-Cap-Ferret, Le Porge, Arcachon, jusqu'a Bordeaux a l'est).

Reutilise effis_fetch.py (bilan saison EFFIS) et firms_fetch.py (hotspots FIRMS
filtres) deja construits pour la carte France entiere, mais restreints a la
bbox du secteur et mis en forme comme des zones (pas des symboles). Le fond de
carte et les routes/forets viennent d'Overpass (OpenStreetMap) et de l'IGN.

Calques (dans cet ordre) :
  fond_departements - limite Gironde/Landes uniquement (data/basemap/departements.geojson), trait fin
  foret             - landuse=forest / natural=wood (Overpass), vert clair
  routes_principales - highway=motorway/trunk/primary (Overpass), gris
  zone_brulee       - union des polygones EFFIS de l'incendie (rose clair)
  zone_active       - union bufferisee des hotspots FIRMS recents (rouge vif)
  villes_labels     - 5 villes reperes, point + texte
  legende
  echelle_nord

Sortie :
  - output/zoom_bassin_arcachon.svg (+ apercu PNG affiche avant validation)
  - output/zone_brulee.geojson (EPSG:4326, proprietes commune/firedate/lastupdate/surface_ha)
  - output/zone_active.geojson (EPSG:4326, proprietes nb_detections/derniere_detection)
  - output/fond_carte_zoom.geojson (EPSG:4326, departements+foret+routes, pour rouvrir
    tout le fond de carte dans QGIS sans dependre du SVG)

Reutilisable : `python scripts/build_zoom_bassin_arcachon.py --date 2026-07-23 --jours-actifs 4`
Un fetch_timestamp unique est capture au debut de l'execution et reutilise partout (logs,
proprietes GeoJSON, comparaison avant/apres) pour que tous les artefacts d'un meme run
soient tracables au meme instant.
"""
import argparse
import json
import time
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import svgwrite
from shapely import concave_hull
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import unary_union

import effis_fetch
import firms_fetch

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
BASEMAP_DEPARTEMENTS = ROOT / "data" / "basemap" / "departements.geojson"
ETAT_PRECEDENT_PATH = ROOT / "data" / "zoom_arcachon_dernier_run.json"
CRS_METRIQUE = "EPSG:2154"  # Lambert-93
CRS_GEOJSON = "EPSG:4326"  # WGS84, standard QGIS/reutilisation - le SVG reste en Lambert-93

# lon_min, lat_min, lon_max, lat_max - nord Bassin d'Arcachon -> Bordeaux a l'est
ZOOM_BBOX = (-1.35, 44.60, -0.55, 44.95)
DEPARTEMENTS_NOMS = ("Gironde", "Landes")
DEPARTEMENTS_CODES = ("33", "40")

VILLES_A_LABELISER = ("Bordeaux", "Arcachon", "La Teste-de-Buch", "Lège-Cap-Ferret", "Le Porge")

SURFACE_MIN_HA_ZONE_BRULEE = 100  # garde-fou : ecarte tout petit feu isole qui tomberait dans la bbox
BUFFER_ACTIF_M = 350  # rayon de buffer autour de chaque hotspot pour former une tache continue
SEUIL_FRAGMENTATION = 3  # nb de composantes disjointes au-dela duquel on demande confirmation

FORET_SIMPLIFY_TOLERANCE_M = 30
ROUTES_SIMPLIFY_TOLERANCE_M = 15

OVERPASS_ENDPOINTS = (
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)

MARGE_PX = 40
LARGEUR_CIBLE_PX = 1400


OVERPASS_HEADERS = {
    # certains miroirs Overpass rejettent le user-agent par defaut de requests
    "User-Agent": "carte-incendies-vdn/1.0 (usage editorial, script interne La Voix du Nord)",
}


def _overpass_query(query, timeout=90):
    """Interroge Overpass en essayant plusieurs miroirs (le serveur principal
    overpass-api.de est frequemment surcharge/indisponible, et certains miroirs
    appliquent un rate-limit court en cas de requetes rapprochees)."""
    derniere_erreur = None
    for url in OVERPASS_ENDPOINTS:
        for tentative in range(2):
            try:
                r = requests.post(url, data=query.encode("utf-8"), headers=OVERPASS_HEADERS, timeout=timeout)
                r.raise_for_status()
                return r.json()["elements"]
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                derniere_erreur = e
                print(f"  [overpass] echec sur {url} ({e!r})"
                      + (", nouvelle tentative" if tentative == 0 else ", miroir suivant"))
                if tentative == 0:
                    time.sleep(5)
    raise RuntimeError(f"Echec Overpass sur tous les miroirs : {derniere_erreur}")


def _bbox_overpass(bbox):
    lon_min, lat_min, lon_max, lat_max = bbox
    return f"{lat_min},{lon_min},{lat_max},{lon_max}"


def fetch_departements(noms=DEPARTEMENTS_NOMS, path=BASEMAP_DEPARTEMENTS, bbox=ZOOM_BBOX):
    """Limite departementale (Gironde/Landes), reprise du shapefile deja utilise pour
    la carte France entiere - juste le contour, pas de polygones communaux. Recadree
    sur la bbox du zoom (les departements entiers sont bien plus grands que le secteur)."""
    gdf = gpd.read_file(path)
    gdf = gdf[gdf["nom"].isin(noms)].to_crs(CRS_METRIQUE)

    minx, miny, maxx, maxy = bbox
    bbox_metrique = (
        gpd.GeoSeries([Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])], crs="EPSG:4326")
        .to_crs(CRS_METRIQUE)
    )
    return gpd.clip(gdf, bbox_metrique)


def fetch_forest(bbox=ZOOM_BBOX, tolerance_m=FORET_SIMPLIFY_TOLERANCE_M):
    bbox_str = _bbox_overpass(bbox)
    query = f"""
    [out:json][timeout:90];
    (
      way["landuse"="forest"]({bbox_str});
      way["natural"="wood"]({bbox_str});
    );
    out geom;
    """
    elements = _overpass_query(query)

    polys = []
    for el in elements:
        coords = [(pt["lon"], pt["lat"]) for pt in el.get("geometry", [])]
        if len(coords) < 4:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = Polygon(coords)
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)
        except Exception:
            continue

    print(f"  [foret] {len(polys)} polygones recuperes via Overpass")
    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=CRS_METRIQUE)

    gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326").to_crs(CRS_METRIQUE)
    gdf["geometry"] = gdf.geometry.simplify(tolerance_m, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf


def fetch_routes(bbox=ZOOM_BBOX, tolerance_m=ROUTES_SIMPLIFY_TOLERANCE_M):
    bbox_str = _bbox_overpass(bbox)
    query = f"""
    [out:json][timeout:90];
    (
      way["highway"~"^(motorway|trunk|primary)$"]({bbox_str});
    );
    out geom;
    """
    elements = _overpass_query(query)

    lines = []
    for el in elements:
        coords = [(pt["lon"], pt["lat"]) for pt in el.get("geometry", [])]
        if len(coords) < 2:
            continue
        lines.append(LineString(coords))

    print(f"  [routes] {len(lines)} troncons recuperes via Overpass")
    if not lines:
        return gpd.GeoDataFrame(geometry=[], crs=CRS_METRIQUE)

    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326").to_crs(CRS_METRIQUE)
    gdf["geometry"] = gdf.geometry.simplify(tolerance_m, preserve_topology=True)
    return gdf


def fetch_villes_labels(noms=VILLES_A_LABELISER, codes_departement=DEPARTEMENTS_CODES, timeout=30):
    """Point 'centre' (bourg) des villes a labeliser, via l'API Decoupage administratif IGN."""
    trouvees = {}
    for code in codes_departement:
        r = requests.get(
            "https://geo.api.gouv.fr/communes",
            params={"codeDepartement": code, "fields": "nom,code,centre"},
            timeout=timeout,
        )
        r.raise_for_status()
        for c in r.json():
            if c["nom"] in noms:
                trouvees[c["nom"]] = c["centre"]["coordinates"]

    manquantes = set(noms) - set(trouvees)
    if manquantes:
        print(f"  ATTENTION : villes non trouvees dans les departements {codes_departement} : {manquantes}")

    gdf = gpd.GeoDataFrame(
        {"nom": list(trouvees.keys())},
        geometry=gpd.points_from_xy(*zip(*trouvees.values())) if trouvees else [],
        crs="EPSG:4326",
    )
    return gdf.to_crs(CRS_METRIQUE)


def fetch_zone_brulee(bbox=ZOOM_BBOX, surface_min_ha=SURFACE_MIN_HA_ZONE_BRULEE, reference_date=None):
    gdf = effis_fetch.fetch_burnt_areas(bbox=bbox)
    if reference_date is not None:
        gdf = gdf[gdf["FIREDATE"].dt.date <= reference_date]
    events = effis_fetch.to_events(gdf)
    cible = events[events["surface_ha"] >= surface_min_ha]

    if cible.empty:
        raise RuntimeError(
            f"Aucun evenement EFFIS >= {surface_min_ha} ha dans la bbox {bbox} : "
            "verifier la bbox ou baisser SURFACE_MIN_HA_ZONE_BRULEE"
        )

    cible_metrique = cible.set_geometry("geometry").to_crs(CRS_METRIQUE)
    geometrie = unary_union(cible_metrique.geometry.values)
    date_debut = cible["date"].min().date()
    surface_totale = cible["surface_ha"].sum()
    print(f"  zone brulee : {len(cible)} polygone(s) EFFIS fusionne(s), "
          f"{surface_totale:.0f} ha au total, depuis le {date_debut.isoformat()}")
    return geometrie, date_debut, cible


def fetch_zone_active(bbox=ZOOM_BBOX, jours=4, reference_date=None, buffer_m=BUFFER_ACTIF_M):
    raw = firms_fetch.fetch_active_fires(bbox=bbox, days=jours, reference_date=reference_date)
    print(f"  {len(raw)} detections FIRMS brutes (secteur zoom, {jours} derniers jours)")
    if raw.empty:
        return None, raw, 0

    filtre_conf = firms_fetch.filter_confidence(raw)
    deduped = firms_fetch.dedupe(filtre_conf)
    filtre_clc = firms_fetch.filter_landcover(deduped)

    if filtre_clc.empty:
        return None, filtre_clc, 0

    gdf = firms_fetch.to_geodataframe(filtre_clc).to_crs(CRS_METRIQUE)
    buffered = gdf.geometry.buffer(buffer_m)
    geometrie = unary_union(buffered.values)

    n_composantes = 1 if geometrie.geom_type == "Polygon" else len(geometrie.geoms)
    print(f"  zone active : {len(gdf)} points bufferises a {buffer_m} m -> {n_composantes} composante(s)")
    return geometrie, gdf, n_composantes


def exporter_zone_brulee_geojson(zone_brulee, cible, fetch_timestamp, path=None):
    """GeoJSON EPSG:4326 (QGIS/reutilisation) - le SVG reste en Lambert-93."""
    path = path or (OUTPUT_DIR / "zone_brulee.geojson")
    idx_max = cible["surface_ha"].idxmax()
    proprietes = {
        "commune": cible.loc[idx_max, "commune"],
        "firedate": cible["date"].min().isoformat(),
        "lastupdate": cible["lastupdate"].max().isoformat(),
        "surface_ha": round(float(cible["surface_ha"].sum()), 1),
        "fetch_timestamp": fetch_timestamp.isoformat(),
    }
    gdf = gpd.GeoDataFrame([proprietes], geometry=[zone_brulee], crs=CRS_METRIQUE).to_crs(CRS_GEOJSON)
    gdf.to_file(path, driver="GeoJSON")
    print(f"Ecrit : {path}")
    return proprietes


def _decouper_en_composantes_avec_membres(geometrie, gdf_points_metrique):
    """Associe a chaque composante disjointe (Polygon) d'une geometrie issue d'un
    buffer+union les points source qui la constituent (chaque point tombe dans
    exactement une composante, celle qu'il a contribue a former)."""
    resultats = []
    for compo in _iter_polygons(geometrie):
        membres = gdf_points_metrique[gdf_points_metrique.geometry.within(compo)]
        resultats.append((compo, membres))
    return resultats


def _proprietes_zone(membres, fetch_timestamp):
    """Agrege les proprietes d'une zone/cluster a partir de ses points membres :
    nb_detections_cluster (somme), frp_max (max), frp_moyen (moyenne ponderee par
    nb de detections par point), derniere_detection (la plus recente)."""
    a_n = "n_detections_cluster" in membres.columns
    nb = int(membres["n_detections_cluster"].sum()) if a_n else len(membres)

    frp_max_zone = None
    if "frp_max" in membres.columns and membres["frp_max"].notna().any():
        frp_max_zone = round(float(membres["frp_max"].max()), 2)

    frp_moyen_zone = None
    if a_n and "frp_moyen" in membres.columns and membres["frp_moyen"].notna().any():
        valides = membres[membres["frp_moyen"].notna()]
        poids = valides["n_detections_cluster"]
        if poids.sum() > 0:
            frp_moyen_zone = round(float((valides["frp_moyen"] * poids).sum() / poids.sum()), 2)

    if "derniere_detection" in membres.columns:
        derniere = membres["derniere_detection"].max()
    else:
        derniere = membres["acq_datetime"].max()

    return {
        "nb_detections_cluster": nb,
        "frp_max": frp_max_zone,
        "frp_moyen": frp_moyen_zone,
        "derniere_detection": derniere.isoformat(),
        "fetch_timestamp": fetch_timestamp.isoformat(),
    }


def _enveloppe_alpha(points_metrique, alpha, buffer_secours_m):
    """Enveloppe concave (shapely concave_hull, ratio=alpha) autour d'un nuage de
    points DEJA regroupes en une composante connexe. alpha=0 -> tres concave (colle
    aux points, respecte les trous/vides reels comme le Bassin d'Arcachon) ; alpha=1
    -> enveloppe convexe (peut combler des trous a tort si alpha trop haut). En
    dessous de 3 points un polygone concave n'a pas de sens : repli sur un petit
    buffer fixe (buffer_secours_m) pour garder une forme affichable."""
    if len(points_metrique) < 3:
        return points_metrique.union_all().buffer(buffer_secours_m)
    hull = concave_hull(MultiPoint(list(points_metrique)), ratio=alpha)
    if hull.geom_type not in ("Polygon", "MultiPolygon") or hull.is_empty:
        return points_metrique.union_all().buffer(buffer_secours_m)
    return hull


def construire_composantes(gdf_points_metrique, buffer_m=BUFFER_ACTIF_M, alpha=None):
    """Regroupe gdf_points_metrique en composantes connexes - le regroupement (QUELS
    points appartiennent a la meme zone) reste base sur buffer+union avec buffer_m,
    inchange. Ce qui change avec `alpha` est la FORME de chaque zone :
      - alpha=None (defaut) : union des buffers, comportement historique ("fleur de
        cercles").
      - alpha in [0, 1] : enveloppe concave (alpha shape) du nuage de points de la
        zone, qui epouse sa forme reelle sans la gonfler en cercles.
    Renvoie une liste de (geometrie, GeoDataFrame des points membres)."""
    if gdf_points_metrique.empty:
        return []

    buffered = gdf_points_metrique.geometry.buffer(buffer_m)
    geometrie_brute = buffered.union_all()

    resultats = []
    for compo in _iter_polygons(geometrie_brute):
        membres = gdf_points_metrique[gdf_points_metrique.geometry.within(compo)]
        if alpha is None:
            forme = compo
        else:
            forme = _enveloppe_alpha(membres.geometry, alpha, buffer_secours_m=buffer_m)
        resultats.append((forme, membres))
    return resultats


def _ecrire_composantes_geojson(composantes, fetch_timestamp, path):
    """Ecrit une liste de (geometrie, membres) sous forme de GeoJSON EPSG:4326, une
    Feature par composante (nb_detections_cluster/frp_max/frp_moyen/derniere_detection
    calcules sur les membres de cette composante). Renvoie la liste des proprietes."""
    if not composantes:
        print(f"  Pas de zone active a exporter (aucun hotspot retenu) : {path} non ecrit")
        return None

    geometries = [c[0] for c in composantes]
    proprietes_liste = [_proprietes_zone(c[1], fetch_timestamp) for c in composantes]

    gdf = gpd.GeoDataFrame(proprietes_liste, geometry=geometries, crs=CRS_METRIQUE).to_crs(CRS_GEOJSON)
    gdf.to_file(path, driver="GeoJSON")
    print(f"Ecrit : {path} ({len(proprietes_liste)} zone(s)/cluster(s))")
    return proprietes_liste


def exporter_points_actifs_geojson(gdf_points_wgs84_ou_metrique, fetch_timestamp, path):
    """GeoJSON EPSG:4326, une Feature Point PAR DETECTION INDIVIDUELLE, sans aucune
    agregation en zones/hulls - pour un rendu heatmap cote frontend, en parallele du
    polygone agrege (zone_active.geojson). Chaque Feature porte lat/lon/frp en
    properties (en plus de la geometrie Point standard, redondant mais pratique pour
    une lib heatmap qui attend des tuples plats plutot qu'un parsing GeoJSON complet)."""
    if gdf_points_wgs84_ou_metrique.empty:
        print(f"  Pas de points actifs a exporter : {path} non ecrit")
        return None

    gdf = gdf_points_wgs84_ou_metrique.to_crs(CRS_GEOJSON)
    proprietes_liste = []
    for _, row in gdf.iterrows():
        proprietes_liste.append({
            "lat": round(row.geometry.y, 5),
            "lon": round(row.geometry.x, 5),
            "frp": float(row["frp"]) if "frp" in row.index and pd.notna(row["frp"]) else None,
            "fetch_timestamp": fetch_timestamp.isoformat(),
        })

    out = gpd.GeoDataFrame(proprietes_liste, geometry=gdf.geometry.values, crs=CRS_GEOJSON)
    out.to_file(path, driver="GeoJSON")
    print(f"Ecrit : {path} ({len(proprietes_liste)} point(s) individuel(s), sans agregation)")
    return proprietes_liste


def exporter_zone_active_geojson(zone_active, hotspots_actifs, fetch_timestamp, path=None):
    """GeoJSON EPSG:4326, une Feature par composante/cluster spatial disjoint (pas un
    seul polygone agrege) : chacune porte ses propres nb_detections_cluster/frp_max/
    frp_moyen/derniere_detection. Renvoie la liste des proprietes (une par zone), ou
    None si pas de zone active (rien a exporter). Comportement historique (union des
    buffers) - voir construire_composantes(..., alpha=...) pour l'enveloppe concave."""
    path = path or (OUTPUT_DIR / "zone_active.geojson")
    if zone_active is None:
        print(f"  Pas de zone active a exporter (aucun hotspot retenu) : {path} non ecrit")
        return None

    composantes = _decouper_en_composantes_avec_membres(zone_active, hotspots_actifs)
    return _ecrire_composantes_geojson(composantes, fetch_timestamp, path)


def exporter_fond_carte_geojson(departements, foret, routes, fetch_timestamp, path=None):
    """Departements + foret + routes en un seul GeoJSON EPSG:4326 (geometries mixtes
    Polygon/LineString - valide en GeoJSON), pour rouvrir tout le fond de carte dans
    QGIS sans redependre du SVG."""
    path = path or (OUTPUT_DIR / "fond_carte_zoom.geojson")

    dep = departements[["nom", "geometry"]].copy()
    dep["type"] = "departement"
    foret_exp = foret[["geometry"]].copy()
    foret_exp["type"] = "foret"
    foret_exp["nom"] = None
    routes_exp = routes[["geometry"]].copy()
    routes_exp["type"] = "route"
    routes_exp["nom"] = None

    fond = pd.concat([dep, foret_exp, routes_exp], ignore_index=True)
    fond["fetch_timestamp"] = fetch_timestamp.isoformat()
    fond = gpd.GeoDataFrame(fond, crs=CRS_METRIQUE).to_crs(CRS_GEOJSON)
    fond.to_file(path, driver="GeoJSON")
    print(f"Ecrit : {path}")


def charger_etat_precedent(path=ETAT_PRECEDENT_PATH):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def sauvegarder_etat(etat, path=ETAT_PRECEDENT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(etat, f, ensure_ascii=False, indent=2)


def afficher_comparaison(etat_precedent, etat_actuel):
    """Resume avant/apres (surfaces, detections, dates de fetch) pour confirmer que la
    mise a jour reflete bien la progression du feu - jamais suppose, toujours affiche."""
    print()
    print("=== Comparaison avec le run precedent ===")
    if etat_precedent is None:
        print("  Aucun run precedent trouve (premiere execution avec suivi d'etat).")
    else:
        print(f"  fetch precedent : {etat_precedent['fetch_timestamp']}")
        print(f"  fetch actuel    : {etat_actuel['fetch_timestamp']}")

        delta_brulee = etat_actuel["zone_brulee_surface_ha"] - etat_precedent["zone_brulee_surface_ha"]
        print(f"  zone brulee : {etat_precedent['zone_brulee_surface_ha']:.0f} ha -> "
              f"{etat_actuel['zone_brulee_surface_ha']:.0f} ha ({delta_brulee:+.0f} ha)")

        avant_det = etat_precedent.get("zone_active_nb_detections")
        apres_det = etat_actuel.get("zone_active_nb_detections")
        if avant_det is not None and apres_det is not None:
            print(f"  zone active : {avant_det} detections -> {apres_det} detections "
                  f"({apres_det - avant_det:+d})")
        print(f"  derniere detection active : {etat_precedent.get('zone_active_derniere_detection')} -> "
              f"{etat_actuel.get('zone_active_derniere_detection')}")
    print()


def _iter_polygons(geom):
    if geom is None:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms


def _iter_lines(geom):
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type == "MultiLineString":
        yield from geom.geoms


def construire_transform(bounds):
    minx, miny, maxx, maxy = bounds
    largeur_m = maxx - minx
    hauteur_m = maxy - miny
    echelle = (LARGEUR_CIBLE_PX - 2 * MARGE_PX) / largeur_m
    largeur_px = LARGEUR_CIBLE_PX
    hauteur_px = hauteur_m * echelle + 2 * MARGE_PX

    def proj(x, y):
        return MARGE_PX + (x - minx) * echelle, MARGE_PX + (maxy - y) * echelle

    return proj, largeur_px, hauteur_px, echelle


def ajouter_echelle_nord(dwg, largeur_px, hauteur_px, echelle):
    """Barre d'echelle (bas gauche) + fleche nord (bas droite). `echelle` en px/metre."""
    g = dwg.g(id="echelle_nord")

    largeur_carte_m = (largeur_px - 2 * MARGE_PX) / echelle
    cible_m = largeur_carte_m * 0.12
    paliers_km = [0.5, 1, 2, 5, 10, 20]
    palier_km = min(paliers_km, key=lambda k: abs(k * 1000 - cible_m))
    barre_px = palier_km * 1000 * echelle

    x0, y0 = MARGE_PX, hauteur_px - 22
    g.add(dwg.line(start=(x0, y0), end=(x0 + barre_px, y0), stroke="#333333", stroke_width=2))
    for x in (x0, x0 + barre_px):
        g.add(dwg.line(start=(x, y0 - 5), end=(x, y0 + 5), stroke="#333333", stroke_width=2))
    g.add(dwg.text(f"{palier_km:g} km", insert=(x0, y0 + 18), font_size=12,
                    font_family="Arial, sans-serif", fill="#333333"))

    cx, cy = largeur_px - MARGE_PX - 10, hauteur_px - 45
    g.add(dwg.polygon(points=[(cx, cy - 22), (cx - 7, cy), (cx + 7, cy)], fill="#333333"))
    g.add(dwg.text("N", insert=(cx - 4, cy + 14), font_size=13, font_family="Arial, sans-serif",
                    font_weight="bold", fill="#333333"))

    return g


def construire_svg(departements, foret, routes, zone_brulee, date_debut, zone_active, villes):
    bounds = departements.total_bounds
    proj, largeur_px, hauteur_px, echelle = construire_transform(bounds)

    dwg = svgwrite.Drawing(size=(f"{largeur_px:.0f}px", f"{hauteur_px:.0f}px"),
                            viewBox=f"0 0 {largeur_px:.0f} {hauteur_px:.0f}",
                            debug=False)
    dwg.add(dwg.rect(insert=(0, 0), size=(largeur_px, hauteur_px), fill="#eaf3f8"))  # fond = mer/bassin

    # --- fond_departements : limite Gironde/Landes uniquement, trait fin
    calque_dep = dwg.g(id="fond_departements")
    for _, row in departements.iterrows():
        for poly in _iter_polygons(row.geometry):
            pts = [proj(x, y) for x, y in poly.exterior.coords]
            calque_dep.add(dwg.polygon(points=pts, fill="none", stroke="#999999", stroke_width=0.8))
    dwg.add(calque_dep)

    # --- foret : vert clair
    calque_foret = dwg.g(id="foret")
    for geom in foret.geometry:
        for poly in _iter_polygons(geom):
            pts = [proj(x, y) for x, y in poly.exterior.coords]
            calque_foret.add(dwg.polygon(points=pts, fill="#c8e0c0", stroke="none"))
    dwg.add(calque_foret)

    # --- routes_principales : gris
    calque_routes = dwg.g(id="routes_principales")
    for geom in routes.geometry:
        for line in _iter_lines(geom):
            pts = [proj(x, y) for x, y in line.coords]
            calque_routes.add(dwg.polyline(points=pts, fill="none", stroke="#8a8a8a", stroke_width=1.3))
    dwg.add(calque_routes)

    # --- zone_brulee : rose clair, opaque (extension cumulee de l'incendie)
    calque_brulee = dwg.g(id="zone_brulee")
    for poly in _iter_polygons(zone_brulee):
        pts = [proj(x, y) for x, y in poly.exterior.coords]
        calque_brulee.add(dwg.polygon(points=pts, fill="#f4b6c2", fill_opacity=0.9,
                                       stroke="#d76b86", stroke_width=1))
    dwg.add(calque_brulee)

    # --- zone_active : contour rouge vif + trame legere (la zone active recouvre en
    # grande partie la zone brulee puisque le feu progresse depuis l'interieur/la
    # lisiere de son propre perimetre ; un remplissage opaque masquerait entierement
    # le rose en dessous, d'ou un remplissage leger + contour marque)
    calque_active = dwg.g(id="zone_active")
    if zone_active is not None:
        for poly in _iter_polygons(zone_active):
            pts = [proj(x, y) for x, y in poly.exterior.coords]
            calque_active.add(dwg.polygon(points=pts, fill="#e30613", fill_opacity=0.4,
                                           stroke="#e30613", stroke_width=2.5))
    dwg.add(calque_active)

    # --- villes_labels : point (losange) + texte, uniquement les villes reperes
    calque_villes = dwg.g(id="villes_labels")
    for _, row in villes.iterrows():
        cx, cy = proj(row.geometry.x, row.geometry.y)
        r = 4
        losange = dwg.polygon(points=[(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
                               fill="#222222")
        calque_villes.add(losange)
        calque_villes.add(dwg.text(row["nom"], insert=(cx + r + 4, cy + 4), font_size=13,
                                    font_family="Arial, sans-serif", fill="#222222"))
    dwg.add(calque_villes)

    # --- legende (zones brulee / active)
    calque_legende = dwg.g(id="legende")
    lx, ly = MARGE_PX, MARGE_PX
    calque_legende.add(dwg.rect(insert=(lx, ly), size=(14, 14), fill="#f4b6c2", fill_opacity=0.9))
    calque_legende.add(dwg.text(f"Zones brulees depuis le {date_debut.strftime('%d/%m/%Y')}",
                                 insert=(lx + 20, ly + 12), font_size=13,
                                 font_family="Arial, sans-serif", fill="#222222"))
    if zone_active is not None:
        calque_legende.add(dwg.rect(insert=(lx, ly + 22), size=(14, 14), fill="#e30613", fill_opacity=0.85))
        calque_legende.add(dwg.text("Feu actif ces derniers jours",
                                     insert=(lx + 20, ly + 34), font_size=13,
                                     font_family="Arial, sans-serif", fill="#222222"))
    dwg.add(calque_legende)

    dwg.add(ajouter_echelle_nord(dwg, largeur_px, hauteur_px, echelle))

    return dwg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="Date de reference (YYYY-MM-DD), defaut = aujourd'hui")
    parser.add_argument("--jours-actifs", type=int, default=4, dest="jours_actifs",
                         help="Fenetre de detections FIRMS pour la zone active (defaut 4 jours)")
    parser.add_argument("--buffer-m", type=float, default=BUFFER_ACTIF_M,
                         help=f"Rayon de buffer (m) autour des hotspots (defaut {BUFFER_ACTIF_M})")
    parser.add_argument("--foret-tolerance-m", type=float, default=FORET_SIMPLIFY_TOLERANCE_M,
                         help=f"Tolerance de simplification des polygones foret (defaut {FORET_SIMPLIFY_TOLERANCE_M} m)")
    args = parser.parse_args()

    reference_date = date_cls.fromisoformat(args.date) if args.date else date_cls.today()
    fetch_timestamp = datetime.now()
    print(f"Date de reference : {reference_date.isoformat()}")
    print(f"Fetch timestamp (unifie pour ce run) : {fetch_timestamp.isoformat()}")

    etat_precedent = charger_etat_precedent()

    print("Chargement de la limite Gironde/Landes...")
    departements = fetch_departements()

    print("Recuperation de la foret (Overpass)...")
    foret = fetch_forest(tolerance_m=args.foret_tolerance_m)

    print("Recuperation des routes principales (Overpass)...")
    routes = fetch_routes()

    print("Recuperation des villes a labeliser...")
    villes = fetch_villes_labels()

    print("Recuperation de la zone brulee (EFFIS)...")
    zone_brulee, date_debut, cible_brulee = fetch_zone_brulee(reference_date=reference_date)

    print("Recuperation de la zone active (FIRMS)...")
    zone_active, hotspots_actifs, n_composantes = fetch_zone_active(
        jours=args.jours_actifs, reference_date=reference_date, buffer_m=args.buffer_m
    )

    if zone_active is not None and n_composantes > SEUIL_FRAGMENTATION:
        print(f"  ATTENTION : {n_composantes} composantes disjointes (> seuil {SEUIL_FRAGMENTATION}) "
              "- la zone active semble fragmentee plutot qu'un front continu.")
        print("  Apercu a verifier avant de choisir : elargir --buffer-m ou --jours-actifs.")

    print("Construction du SVG...")
    dwg = construire_svg(departements, foret, routes, zone_brulee, date_debut, zone_active, villes)

    OUTPUT_DIR.mkdir(exist_ok=True)
    svg_path = OUTPUT_DIR / "zoom_bassin_arcachon.svg"
    dwg.saveas(str(svg_path))
    print(f"Ecrit : {svg_path}")

    print("Export GeoJSON (EPSG:4326)...")
    proprietes_brulee = exporter_zone_brulee_geojson(zone_brulee, cible_brulee, fetch_timestamp)
    proprietes_active_liste = exporter_zone_active_geojson(zone_active, hotspots_actifs, fetch_timestamp)
    exporter_fond_carte_geojson(departements, foret, routes, fetch_timestamp)

    # une Feature par zone/cluster desormais - on agrege sur toutes les zones pour le
    # resume avant/apres (nb total de detections, frp max toutes zones confondues, etc.)
    nb_detections_total = None
    derniere_detection_globale = None
    frp_max_global = None
    if proprietes_active_liste:
        nb_detections_total = sum(p["nb_detections_cluster"] for p in proprietes_active_liste)
        derniere_detection_globale = max(p["derniere_detection"] for p in proprietes_active_liste)
        frp_values = [p["frp_max"] for p in proprietes_active_liste if p["frp_max"] is not None]
        frp_max_global = max(frp_values) if frp_values else None

    etat_actuel = {
        "fetch_timestamp": fetch_timestamp.isoformat(),
        "reference_date": reference_date.isoformat(),
        "zone_brulee_surface_ha": proprietes_brulee["surface_ha"],
        "zone_brulee_firedate": proprietes_brulee["firedate"],
        "zone_brulee_lastupdate": proprietes_brulee["lastupdate"],
        "zone_active_nb_detections": nb_detections_total,
        "zone_active_derniere_detection": derniere_detection_globale,
        "zone_active_frp_max": frp_max_global,
        "zone_active_n_composantes": n_composantes,
    }
    afficher_comparaison(etat_precedent, etat_actuel)
    sauvegarder_etat(etat_actuel)

    return svg_path, n_composantes


if __name__ == "__main__":
    main()
