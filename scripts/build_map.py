"""Construction de la carte SVG des incendies de la saison en France.

Deux jeux de donnees :
  - bilan saison (EFFIS WFS, modis.ba.poly.season) -> carres proportionnels
    a la surface brulee, "gros carre" si > 1000 ha.
  - feux actifs maintenant (FIRMS VIIRS NOAA20 + SNPP, dernieres `--jours`
    journees) -> points.

Sortie :
  - output/carte_incendies_france.svg (calques fond_carte / petits_carres /
    gros_carres / points_actifs, pas de texte ni de lignes de rappel, sauf le
    calque legende_points_actifs qui porte une legende textuelle dynamique)
  - output/labels_gros_feux.csv (position + libelle des gros feux, pour le
    montage manuel dans Illustrator)
  - output/points_actifs_complet.csv (un point par cluster de detections
    actives, toutes les colonnes, TOUS les clusters - reference/verification,
    pas pour la labellisation)
  - output/labels_points_actifs.csv (uniquement les --top-n-actifs plus gros
    clusters, memes colonnes, pour le placement manuel des noms dans
    Illustrator - cf. "5 feux les plus importants" du modele Le Monde)

Reutilisable : `python scripts/build_map.py --date 2026-07-23 --top-n-actifs 5`
(par defaut, la date du jour, top 5).
"""
import argparse
import concurrent.futures
import json
import math
import re
import time
from datetime import date as date_cls
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import svgwrite

import effis_fetch
import firms_fetch

ROOT = Path(__file__).resolve().parent.parent
BASEMAP_DEPARTEMENTS = ROOT / "data" / "basemap" / "departements.geojson"
BASEMAP_REGIONS = ROOT / "data" / "basemap" / "regions.geojson"
OUTPUT_DIR = ROOT / "output"

CRS_METRIQUE = "EPSG:2154"  # Lambert-93

MOIS_FR = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai", 6: "juin",
    7: "juillet", 8: "août", 9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre",
}

# Reverse-geocoding communal (point -> commune/departement) pour labels_points_actifs.csv,
# via l'API Decoupage administratif de l'IGN (un point = une requete legere, pas besoin
# de charger un shapefile communes national juste pour ~quelques centaines de points).
# Le nom de departement est ensuite resolu via le shapefile departements deja charge
# pour le fond de carte (code -> nom), donc pas de deuxieme appel reseau pour ca.
COMMUNE_API_URL = "https://geo.api.gouv.fr/communes"
COMMUNE_CACHE_PATH = ROOT / "data" / "commune_cache.json"
COMMUNE_CACHE_DECIMALES = 3
COMMUNE_MAX_WORKERS = 10

# Rouge unique pour les deux cartes statiques (incendies depuis le 1er janvier /
# incendies en cours, derniers 48h) - meme couleur de marque sur les deux visuels,
# la distinction se fait par le calque (petits_carres/gros_carres vs points_actifs)
# et le titre/legende ajoutes dans Illustrator, pas par la couleur.
ROUGE_INCENDIES = "#e7412b"

# --- Mise a l'echelle des carres (symboles proportionnels : cote ~ sqrt(surface)
# pour que ce soit la SURFACE du symbole qui soit proportionnelle a la surface
# brulee, convention standard des cartes a symboles proportionnels). Valeurs
# choisies pour un rendu lisible entre le plus petit et le plus gros feu de la
# saison ; a ajuster librement dans build_map.py si besoin.
SCALE_K = 0.6
MIN_SIDE_PX = 2.5

# Points actifs FIRMS : carre de meme famille visuelle que les carres du bilan
# saison, taille legerement variable selon le nombre de detections brutes
# fusionnees dans le cluster - variation subtile, plafonnee, pour rester lisible
# et ne jamais rivaliser visuellement avec les gros carres.
HOTSPOT_MIN_SIDE_PX = 3
HOTSPOT_MAX_SIDE_PX = 9
HOTSPOT_SCALE_K = 0.6
HOTSPOT_COLOR = ROUGE_INCENDIES

MARGE_PX = 20
LARGEUR_CIBLE_PX = 1000


def cote_carre(surface_ha):
    return max(MIN_SIDE_PX, SCALE_K * math.sqrt(surface_ha))


def cote_hotspot(nb_detections_cluster):
    cote = HOTSPOT_MIN_SIDE_PX + HOTSPOT_SCALE_K * math.sqrt(max(0, nb_detections_cluster - 1))
    return min(HOTSPOT_MAX_SIDE_PX, cote)


def legende_points_actifs_texte(reference_date):
    return f"Détections actives au {reference_date.day} {MOIS_FR[reference_date.month]} {reference_date.year}"


def charger_fond_carte():
    departements = gpd.read_file(BASEMAP_DEPARTEMENTS).to_crs(CRS_METRIQUE)
    regions = gpd.read_file(BASEMAP_REGIONS).to_crs(CRS_METRIQUE)
    return departements, regions


def construire_transform(bounds):
    minx, miny, maxx, maxy = bounds
    largeur_m = maxx - minx
    hauteur_m = maxy - miny
    echelle = (LARGEUR_CIBLE_PX - 2 * MARGE_PX) / largeur_m
    largeur_px = LARGEUR_CIBLE_PX
    hauteur_px = hauteur_m * echelle + 2 * MARGE_PX

    def proj(x, y):
        svg_x = MARGE_PX + (x - minx) * echelle
        svg_y = MARGE_PX + (maxy - y) * echelle  # inversion de l'axe Y
        return svg_x, svg_y

    return proj, largeur_px, hauteur_px


def ajouter_fond_carte(dwg, groupe, departements, regions):
    g_dep = dwg.g(id="fond_carte-departements")
    for geom in departements.geometry:
        for poly in _iter_polygons(geom):
            pts = [dwg.proj(x, y) for x, y in poly.exterior.coords]
            g_dep.add(dwg.polygon(points=pts, fill="#e6e6e6", stroke="#ffffff", stroke_width=0.5))
    groupe.add(g_dep)

    g_reg = dwg.g(id="fond_carte-regions")
    for geom in regions.geometry:
        for poly in _iter_polygons(geom):
            pts = [dwg.proj(x, y) for x, y in poly.exterior.coords]
            g_reg.add(dwg.polygon(points=pts, fill="none", stroke="#999999", stroke_width=1))
    groupe.add(g_reg)


def _iter_polygons(geom):
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms


def construire_evenements_saison(reference_date=None):
    gdf = effis_fetch.fetch_burnt_areas()
    if reference_date is not None:
        gdf = gdf[gdf["FIREDATE"].dt.date <= reference_date]
    events = effis_fetch.to_events(gdf)
    events = effis_fetch.merge_adjacent_events(events)
    events = effis_fetch.classify_events(events)
    events = events.set_geometry("geometry").to_crs(CRS_METRIQUE)
    events["centroid"] = events.geometry.centroid
    return events


def construire_points_actifs(reference_date=None, jours=2):
    raw = firms_fetch.fetch_active_fires(days=jours, reference_date=reference_date)
    print(f"  {len(raw)} detections brutes")
    filtre_conf = firms_fetch.filter_confidence(raw)
    deduped = firms_fetch.dedupe(filtre_conf)
    filtre_clc = firms_fetch.filter_landcover(deduped)
    gdf = firms_fetch.to_geodataframe(filtre_clc).to_crs(CRS_METRIQUE)
    return gdf


def clip_hotspots_to_france(hotspots, regions):
    """Ecarte strictement (sans buffer) les points hors du polygone dissous des
    regions metropolitaines - ex. detections en Belgique/Espagne/Suisse/Manche
    presentes dans la bbox de requete FIRMS mais hors territoire francais."""
    france_polygon = regions.geometry.union_all()
    avant = len(hotspots)
    out = hotspots[hotspots.within(france_polygon)].reset_index(drop=True)
    print(f"  [clip France metropolitaine] {avant} -> {len(out)} ({avant - len(out)} points hors territoire ecartes)")
    return out


def _load_json_cache(path):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json_cache(cache, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def _commune_cache_key(lat, lon, decimales=COMMUNE_CACHE_DECIMALES):
    return f"{round(lat, decimales)},{round(lon, decimales)}"


def _query_commune(lat, lon, timeout=15, tentatives=3):
    """Reverse-geocoding communal via geo.api.gouv.fr (point -> commune + code departement).
    Fail-open : renvoie (None, None) si toutes les tentatives echouent, plutot que de planter -
    labels_points_actifs.csv aura juste une ligne avec commune/departement vides a completer."""
    for tentative in range(tentatives):
        try:
            r = requests.get(
                COMMUNE_API_URL,
                params={"lat": lat, "lon": lon, "fields": "nom,codeDepartement", "limit": 1},
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            if data:
                return data[0]["nom"], data[0]["codeDepartement"]
            return None, None
        except (requests.exceptions.RequestException, ValueError, KeyError, IndexError):
            if tentative < tentatives - 1:
                time.sleep(1)
    return None, None


def construire_labels_points_actifs(hotspots, departements, proj, cache_path=COMMUNE_CACHE_PATH,
                                     max_workers=COMMUNE_MAX_WORKERS):
    """Un point par cluster de detections actives (deja dedoublonnees dans
    construire_points_actifs) : commune/departement par reverse-geocoding, derniere
    detection du cluster, position SVG - pour reperage manuel dans Illustrator."""
    colonnes = ["commune", "departement", "date_derniere_detection", "nb_detections_cluster", "x_svg", "y_svg"]
    if hotspots.empty:
        return pd.DataFrame(columns=colonnes)

    dep_par_code = dict(zip(departements["code"], departements["nom"]))
    cache = _load_json_cache(cache_path)
    cache_modifie = False

    resultats = [None] * len(hotspots)
    a_interroger = []
    for i, row in enumerate(hotspots.itertuples()):
        key = _commune_cache_key(row.latitude, row.longitude)
        if key in cache:
            resultats[i] = tuple(cache[key])
        else:
            a_interroger.append((i, key, row.latitude, row.longitude))

    print(f"  [communes points actifs] {len(hotspots) - len(a_interroger)} depuis le cache local "
          f"({cache_path.name}), {len(a_interroger)} a interroger en direct")

    if a_interroger:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_query_commune, lat, lon): (i, key)
                for i, key, lat, lon in a_interroger
            }
            for future in concurrent.futures.as_completed(futures):
                i, key = futures[future]
                nom, code_dep = future.result()
                resultats[i] = (nom, code_dep)
                cache[key] = [nom, code_dep]
                cache_modifie = True

    if cache_modifie:
        _save_json_cache(cache, cache_path)

    lignes = []
    for i, row in enumerate(hotspots.itertuples()):
        nom, code_dep = resultats[i]
        cx, cy = proj(row.geometry.x, row.geometry.y)
        lignes.append({
            "commune": nom,
            "departement": dep_par_code.get(code_dep, code_dep),
            "date_derniere_detection": row.derniere_detection.strftime("%d/%m/%Y %H:%M"),
            "nb_detections_cluster": int(row.n_detections_cluster),
            "x_svg": round(cx, 1),
            "y_svg": round(cy, 1),
        })

    return pd.DataFrame(lignes, columns=colonnes)


def construire_svg(departements, regions, events, hotspots, reference_date):
    bounds = departements.total_bounds
    proj, largeur_px, hauteur_px = construire_transform(bounds)

    dwg = svgwrite.Drawing(size=(f"{largeur_px:.0f}px", f"{hauteur_px:.0f}px"),
                            viewBox=f"0 0 {largeur_px:.0f} {hauteur_px:.0f}",
                            debug=False)  # necessaire pour autoriser les attributs data-*
    dwg.proj = proj  # petit raccourci pratique

    calque_fond = dwg.g(id="fond_carte")
    ajouter_fond_carte(dwg, calque_fond, departements, regions)
    dwg.add(calque_fond)

    calque_petits = dwg.g(id="petits_carres")
    calque_gros = dwg.g(id="gros_carres")

    labels_rows = []
    for _, row in events.iterrows():
        cx, cy = proj(row["centroid"].x, row["centroid"].y)
        side = cote_carre(row["surface_ha"])
        rect = dwg.rect(
            insert=(cx - side / 2, cy - side / 2),
            size=(side, side),
            fill=ROUGE_INCENDIES,
        )
        rect["data-commune"] = row["commune"]
        rect["data-departement"] = row["departement"]
        rect["data-date"] = row["date"].date().isoformat()
        rect["data-surface_ha"] = f"{row['surface_ha']:.0f}"

        if row["classe"] == "gros_carre":
            label = f"{row['commune']} ({row['departement']}) : {row['surface_ha']:.0f} ha"
            rect["data-label"] = label
            calque_gros.add(rect)
            labels_rows.append({
                "commune": row["commune"],
                "departement": row["departement"],
                "date": row["date"].date().isoformat(),
                "surface_ha": row["surface_ha"],
                "label": label,
                "x_svg": round(cx, 1),
                "y_svg": round(cy, 1),
            })
        else:
            calque_petits.add(rect)

    dwg.add(calque_petits)
    dwg.add(calque_gros)

    dwg.add(construire_calque_points_actifs(dwg, hotspots, proj))
    dwg.add(construire_calque_legende_actifs(dwg, reference_date))

    return dwg, pd.DataFrame(labels_rows), proj


def construire_calque_points_actifs(dwg, hotspots, proj):
    calque_actifs = dwg.g(id="points_actifs")
    for _, row in hotspots.iterrows():
        cx, cy = proj(row.geometry.x, row.geometry.y)
        nb = int(row["n_detections_cluster"]) if "n_detections_cluster" in row.index else 1
        side = cote_hotspot(nb)
        rect = dwg.rect(
            insert=(cx - side / 2, cy - side / 2),
            size=(side, side),
            fill=HOTSPOT_COLOR,
        )
        rect["data-date"] = row["acq_date"]
        rect["data-heure"] = str(row["acq_time"])
        rect["data-confidence"] = str(row["confidence"])
        rect["data-capteur"] = row["capteur"]
        rect["data-nb_detections_cluster"] = str(nb)
        if "clc_code" in row.index:
            rect["data-clc_code"] = str(row["clc_code"])
        calque_actifs.add(rect)
    return calque_actifs


def construire_calque_legende_actifs(dwg, reference_date):
    calque_legende_actifs = dwg.g(id="legende_points_actifs")
    lx, ly = MARGE_PX, MARGE_PX
    calque_legende_actifs.add(dwg.rect(insert=(lx, ly), size=(10, 10), fill=HOTSPOT_COLOR))
    calque_legende_actifs.add(dwg.text(
        legende_points_actifs_texte(reference_date),
        insert=(lx + 16, ly + 9), font_size=11, font_family="Arial, sans-serif", fill="#222222",
    ))
    return calque_legende_actifs


def actualiser_points_actifs_seulement(departements, regions, reference_date, jours, top_n_actifs):
    """Reconstruit uniquement les calques points_actifs/legende_points_actifs dans le
    SVG existant, sans toucher au bilan saison (EFFIS) deja en place - utilise quand EFFIS
    est indisponible mais qu'on veut quand meme rafraichir la carte "incendies en cours" (48h)."""
    svg_path = OUTPUT_DIR / "carte_incendies_france.svg"
    csv_actifs_complet_path = OUTPUT_DIR / "points_actifs_complet.csv"
    csv_actifs_labels_path = OUTPUT_DIR / "labels_points_actifs.csv"

    if not svg_path.exists():
        raise FileNotFoundError(
            f"{svg_path} n'existe pas encore - un run complet (avec EFFIS) est necessaire "
            "avant de pouvoir n'actualiser que le calque points_actifs."
        )

    print("Recuperation feux actifs FIRMS...")
    hotspots = construire_points_actifs(reference_date=reference_date, jours=jours)
    hotspots = clip_hotspots_to_france(hotspots, regions)
    print(f"  {len(hotspots)} detections actives retenues (apres tous filtres + clip)")

    # meme transform que construire_svg (deterministe, base uniquement sur les bornes du
    # fond de carte communal/regional - inchange, donc coordonnees alignees avec le SVG existant)
    proj, _, _ = construire_transform(departements.total_bounds)
    dwg = svgwrite.Drawing(debug=False)
    dwg.proj = proj

    print("Reconstruction du calque points_actifs (bilan EFFIS existant conserve tel quel)...")
    fragment_actifs = construire_calque_points_actifs(dwg, hotspots, proj).tostring()
    fragment_legende = construire_calque_legende_actifs(dwg, reference_date).tostring()

    svg_texte = svg_path.read_text(encoding="utf-8")
    svg_texte, n1 = re.subn(r'<g id="points_actifs">.*?</g>', lambda m: fragment_actifs,
                             svg_texte, count=1, flags=re.DOTALL)
    svg_texte, n2 = re.subn(r'<g id="legende_points_actifs">.*?</g>', lambda m: fragment_legende,
                             svg_texte, count=1, flags=re.DOTALL)
    if n1 == 0 or n2 == 0:
        raise RuntimeError(
            f"Calque(s) introuvable(s) dans {svg_path} (points_actifs: {n1}, "
            f"legende_points_actifs: {n2}) - le SVG existant ne correspond peut-etre pas "
            "au format attendu."
        )
    svg_path.write_text(svg_texte, encoding="utf-8")
    print(f"Ecrit (calque points_actifs uniquement) : {svg_path}")

    print("Construction des labels points actifs (reverse-geocoding communal)...")
    points_actifs_complet_df = construire_labels_points_actifs(hotspots, departements, proj)
    points_actifs_complet_df = points_actifs_complet_df.sort_values("nb_detections_cluster", ascending=False)
    labels_actifs_df = points_actifs_complet_df.head(top_n_actifs)

    points_actifs_complet_df.to_csv(csv_actifs_complet_path, index=False)
    labels_actifs_df.to_csv(csv_actifs_labels_path, index=False)
    print(f"Ecrit : {csv_actifs_complet_path} ({len(points_actifs_complet_df)} lignes)")
    print(f"Ecrit : {csv_actifs_labels_path} ({len(labels_actifs_df)} lignes, top {top_n_actifs})")
    print("Bilan saison (EFFIS) NON touche - petits_carres/gros_carres/labels_gros_feux.csv inchanges.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None,
                         help="Date de reference (YYYY-MM-DD), defaut = aujourd'hui")
    parser.add_argument("--jours", type=int, default=2,
                         help="Nombre de jours de detections FIRMS a recuperer (defaut 2)")
    parser.add_argument("--top-n-actifs", type=int, default=5, dest="top_n_actifs",
                         help="Nombre de clusters actifs a labelliser dans Illustrator, "
                              "classes par nb_detections_cluster decroissant (defaut 5)")
    parser.add_argument("--seulement-actifs", action="store_true", dest="seulement_actifs",
                         help="N'actualise que le calque points_actifs (48h, FIRMS) dans le "
                              "SVG existant, sans toucher au bilan saison (EFFIS) deja present "
                              "- utile quand EFFIS est indisponible. Le fichier SVG de sortie "
                              "doit deja exister (issu d'un run complet precedent).")
    args = parser.parse_args()

    reference_date = date_cls.fromisoformat(args.date) if args.date else date_cls.today()

    print(f"Date de reference : {reference_date.isoformat()}")

    print("Chargement du fond de carte...")
    departements, regions = charger_fond_carte()

    if args.seulement_actifs:
        actualiser_points_actifs_seulement(
            departements, regions, reference_date, jours=args.jours, top_n_actifs=args.top_n_actifs,
        )
        return

    print("Recuperation bilan saison EFFIS...")
    events = construire_evenements_saison(reference_date=reference_date)
    print(f"  {len(events)} evenements ({(events['classe'] == 'gros_carre').sum()} gros feux)")

    print("Recuperation feux actifs FIRMS...")
    hotspots = construire_points_actifs(reference_date=reference_date, jours=args.jours)
    hotspots = clip_hotspots_to_france(hotspots, regions)
    print(f"  {len(hotspots)} detections actives retenues (apres tous filtres + clip)")

    print("Construction du SVG...")
    dwg, labels_df, proj = construire_svg(departements, regions, events, hotspots, reference_date)

    print("Construction des labels points actifs (reverse-geocoding communal)...")
    points_actifs_complet_df = construire_labels_points_actifs(hotspots, departements, proj)
    points_actifs_complet_df = points_actifs_complet_df.sort_values("nb_detections_cluster", ascending=False)

    # points_actifs_complet.csv : les 385 (ou N) clusters, sert au dessin des carres (deja
    # fait directement depuis `hotspots` dans construire_svg) - pas a la labellisation.
    # labels_points_actifs.csv : uniquement les --top-n-actifs plus gros clusters, a utiliser
    # pour le placement manuel des noms dans Illustrator (cf. "5 feux les plus importants" Le Monde).
    labels_actifs_df = points_actifs_complet_df.head(args.top_n_actifs)

    OUTPUT_DIR.mkdir(exist_ok=True)
    svg_path = OUTPUT_DIR / "carte_incendies_france.svg"
    csv_path = OUTPUT_DIR / "labels_gros_feux.csv"
    csv_actifs_complet_path = OUTPUT_DIR / "points_actifs_complet.csv"
    csv_actifs_labels_path = OUTPUT_DIR / "labels_points_actifs.csv"

    dwg.saveas(str(svg_path))
    labels_df.sort_values("surface_ha", ascending=False).to_csv(csv_path, index=False)
    points_actifs_complet_df.to_csv(csv_actifs_complet_path, index=False)
    labels_actifs_df.to_csv(csv_actifs_labels_path, index=False)

    print(f"Ecrit : {svg_path}")
    print(f"Ecrit : {csv_path}")
    print(f"Ecrit : {csv_actifs_complet_path} ({len(points_actifs_complet_df)} lignes)")
    print(f"Ecrit : {csv_actifs_labels_path} ({len(labels_actifs_df)} lignes, top {args.top_n_actifs})")


if __name__ == "__main__":
    main()
