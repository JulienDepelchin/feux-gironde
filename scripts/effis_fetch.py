"""Recuperation du bilan saison des feux de foret (EFFIS WFS) pour la France."""
import os
import tempfile
import time
import zipfile

import geopandas as gpd
import pandas as pd
import requests

EFFIS_WFS_URL = "https://maps.effis.emergency.copernicus.eu/effis"
FRANCE_BBOX = (-5.2, 41.3, 9.6, 51.1)  # lon_min, lat_min, lon_max, lat_max (metropole + Corse)
SEUIL_GROS_FEU_HA = 1000

CRS_METRIQUE_FUSION = "EPSG:2154"  # Lambert-93, pour les calculs de distance
DISTANCE_FUSION_DEFAUT_M = 1500
JOURS_FUSION_DEFAUT = 7

# Le chainage transitif (union-find) peut relier une longue serie d'evenements
# proches deux a deux mais tres eloignes globalement (ex. une vague de feux
# pastoraux distincts sur plusieurs jours dans toute une chaine de montagne).
# On plafonne donc l'ecart TOTAL du groupe (pas seulement chaque paire) : au-dela,
# le groupe est rejete (non fusionne, loggue pour verification manuelle).
DIAMETRE_MAX_FACTEUR = 3  # etendue spatiale max autorisee = distance_fusion_m * ce facteur

# Communes a surveiller specifiquement : si l'ecart date_min/date_max d'un evenement
# fusionne les concernant depasse ce seuil, c'est un signe que FIREDATE est une date
# de suivi/mise a jour (et non de depart du feu) - a signaler explicitement.
SEUIL_ALERTE_ECART_JOURS = 2
COMMUNES_A_SURVEILLER = ("Porge", "Noisy-sur-École")

# Le serveur WFS d'EFFIS est parfois tres lent a repondre (30-90s+) et
# ponctuellement injoignable ; on retente avant d'abandonner.
MAX_TENTATIVES = 4
DELAI_ENTRE_TENTATIVES_S = 15


def fetch_burnt_areas(bbox=FRANCE_BBOX, country="FR", timeout=180):
    """Telecharge les polygones de la saison en cours via WFS et retourne un GeoDataFrame filtre pays."""
    params = {
        "service": "WFS",
        "request": "getfeature",
        "typename": "ms:modis.ba.poly.season",
        "version": "1.1.0",
        "outputformat": "SHAPEZIP",
        "bbox": ",".join(str(v) for v in bbox),
    }

    derniere_erreur = None
    r = None
    for tentative in range(1, MAX_TENTATIVES + 1):
        try:
            r = requests.get(EFFIS_WFS_URL, params=params, timeout=timeout)
            r.raise_for_status()
            break
        except (requests.exceptions.RequestException,) as e:
            derniere_erreur = e
            print(f"  [effis] tentative {tentative}/{MAX_TENTATIVES} echouee ({e!r})")
            if tentative < MAX_TENTATIVES:
                time.sleep(DELAI_ENTRE_TENTATIVES_S)
    else:
        raise RuntimeError(
            f"Echec de recuperation EFFIS apres {MAX_TENTATIVES} tentatives"
        ) from derniere_erreur

    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, "ba.zip")
        with open(zpath, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        shp_path = os.path.join(tmp, "modis.ba.poly.season.shp")
        gdf = gpd.read_file(shp_path, encoding="cp1252")

    gdf["AREA_HA"] = gdf["AREA_HA"].astype(float)
    gdf["FIREDATE"] = pd.to_datetime(gdf["FIREDATE"], format="mixed")
    gdf["LASTUPDATE"] = pd.to_datetime(gdf["LASTUPDATE"], format="mixed")
    if country:
        gdf = gdf[gdf["COUNTRY"] == country].copy()
    return gdf


def to_events(gdf):
    """Reduit le GeoDataFrame de polygones a une table d'evenements (geometrie complete,
    pas encore de centroide). Pas de classement gros/petit carre ici : voir classify_events(),
    a appliquer APRES merge_adjacent_events()."""
    out = gdf[["FIREDATE", "LASTUPDATE", "PROVINCE", "COMMUNE", "AREA_HA", "geometry"]].copy()
    out = out.rename(columns={
        "FIREDATE": "date",
        "LASTUPDATE": "lastupdate",
        "PROVINCE": "departement",
        "COMMUNE": "commune",
        "AREA_HA": "surface_ha",
    })
    return out.reset_index(drop=True)


def classify_events(events, seuil_ha=SEUIL_GROS_FEU_HA):
    """Ajoute la colonne 'classe' (gros_carre si > seuil_ha, sinon petit_carre).
    A appeler APRES merge_adjacent_events() pour que le seuil s'applique aux
    evenements deja fusionnes, pas aux fragments individuels."""
    events = events.copy()
    events["classe"] = events["surface_ha"].apply(
        lambda ha: "gros_carre" if ha > seuil_ha else "petit_carre"
    )
    return events


def _grouper_par_proximite(events, distance_fusion_m, jours_fusion):
    """Union-Find : relie deux evenements si leurs geometries sont a moins de
    distance_fusion_m ET leurs dates a moins de jours_fusion jours d'ecart.
    Renvoie la liste des groupes (listes d'indices positionnels dans `events`)."""
    n = len(events)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    metrique = events.set_geometry("geometry").to_crs(CRS_METRIQUE_FUSION)
    buffered = gpd.GeoDataFrame(geometry=metrique.geometry.buffer(distance_fusion_m), crs=CRS_METRIQUE_FUSION)
    original = gpd.GeoDataFrame(geometry=metrique.geometry, crs=CRS_METRIQUE_FUSION)

    # buffer(distance_fusion_m) puis intersects <=> distance <= distance_fusion_m, exactement
    paires = gpd.sjoin(buffered, original, predicate="intersects", how="inner")

    dates = events["date"].values
    for i, j in zip(paires.index, paires["index_right"]):
        if i == j:
            continue
        ecart_jours = abs((dates[i] - dates[j]) / pd.Timedelta(days=1))
        if ecart_jours <= jours_fusion:
            union(i, j)

    groupes = {}
    for idx in range(n):
        groupes.setdefault(find(idx), []).append(idx)
    return list(groupes.values())


def merge_adjacent_events(events, distance_fusion_m=DISTANCE_FUSION_DEFAUT_M, jours_fusion=JOURS_FUSION_DEFAUT,
                           diametre_max_facteur=DIAMETRE_MAX_FACTEUR, verbose=True):
    """Regroupe les evenements EFFIS proches dans l'espace ET le temps (probablement le
    meme incendie fragmente en plusieurs polygones). Deux evenements sont relies si leurs
    geometries sont a moins de `distance_fusion_m` metres l'une de l'autre ET si leurs
    dates sont a moins de `jours_fusion` jours d'ecart.

    Le chainage transitif (A-B et B-C relies fusionnent A/B/C meme si A-C ne l'est pas
    directement) peut regrouper une longue serie d'evenements distincts (ex. une vague de
    feux pastoraux sur plusieurs jours dans toute une chaine de montagne). Pour l'eviter,
    un groupe n'est fusionne QUE SI son etendue totale reste raisonnable : ecart de dates
    (max-min) <= jours_fusion ET diametre spatial (diagonale de l'emprise) <=
    distance_fusion_m * diametre_max_facteur. Sinon le groupe est REJETE (les evenements
    restent individuels) et loggue pour verification manuelle - jamais fusionne a l'aveugle.

    Pour chaque groupe accepte : surface_ha = somme, date = la plus ancienne, geometrie =
    union des polygones, commune/departement = ceux de l'evenement du groupe ayant la plus
    grande surface individuelle.
    """
    if events.empty:
        return events

    events = events.reset_index(drop=True)
    groupes = _grouper_par_proximite(events, distance_fusion_m, jours_fusion)
    diametre_max_m = distance_fusion_m * diametre_max_facteur

    lignes = []
    n_fusions = 0
    n_rejets = 0
    for groupe in groupes:
        if len(groupe) == 1:
            ligne = events.iloc[groupe[0]].to_dict()
            ligne["date_min"] = ligne["date"]
            ligne["date_max"] = ligne["date"]
            lignes.append(ligne)
            continue

        sous = events.iloc[groupe]
        ecart_jours = (sous["date"].max() - sous["date"].min()) / pd.Timedelta(days=1)
        diametre_m = sous.set_geometry("geometry").to_crs(CRS_METRIQUE_FUSION).total_bounds
        diametre_m = ((diametre_m[2] - diametre_m[0]) ** 2 + (diametre_m[3] - diametre_m[1]) ** 2) ** 0.5

        if ecart_jours > jours_fusion or diametre_m > diametre_max_m:
            n_rejets += 1
            if verbose:
                print(f"  [fusion REJETEE #{n_rejets}] {len(groupe)} evenements chaines mais "
                      f"etendue totale excessive (ecart {ecart_jours:.1f} j, diametre {diametre_m:.0f} m) "
                      "- conserves individuellement, a verifier manuellement :")
                for _, row in sous.sort_values("date").iterrows():
                    print(f"      {row['date'].date().isoformat()}  {row['commune']:<30s} "
                          f"({row['departement']:<25s})  {row['surface_ha']:.0f} ha")
            for idx in groupe:
                ligne = events.iloc[idx].to_dict()
                ligne["date_min"] = ligne["date"]
                ligne["date_max"] = ligne["date"]
                lignes.append(ligne)
            continue

        n_fusions += 1
        idx_max = sous["surface_ha"].idxmax()
        ligne_ref = events.loc[idx_max]
        surface_totale = sous["surface_ha"].sum()
        date_min = sous["date"].min()
        date_max = sous["date"].max()

        if verbose:
            print(f"  [fusion #{n_fusions}] {len(groupe)} evenements regroupes "
                  f"(<= {distance_fusion_m:.0f} m, <= {jours_fusion} j) -> "
                  f"{ligne_ref['commune']} ({ligne_ref['departement']}), {surface_totale:.0f} ha au total, "
                  f"du {date_min.date().isoformat()} au {date_max.date().isoformat()} :")
            for _, row in sous.sort_values("date").iterrows():
                print(f"      {row['date'].date().isoformat()}  {row['commune']:<30s} "
                      f"({row['departement']:<25s})  {row['surface_ha']:.0f} ha")

        lignes.append({
            "date": date_min,
            "date_min": date_min,
            "date_max": date_max,
            "departement": ligne_ref["departement"],
            "commune": ligne_ref["commune"],
            "surface_ha": surface_totale,
            "geometry": sous.geometry.union_all(),
        })

    if verbose:
        print(f"  [fusion] {len(events)} evenements -> {len(lignes)} apres fusion "
              f"({n_fusions} groupe(s) fusionne(s), {n_rejets} groupe(s) rejete(s) pour etendue excessive)")

    resultat = gpd.GeoDataFrame(lignes, geometry="geometry", crs=events.crs).reset_index(drop=True)

    if verbose:
        _alerter_ecart_dates_suivi(resultat)

    return resultat


def _alerter_ecart_dates_suivi(events_fusionnes, seuil_jours=SEUIL_ALERTE_ECART_JOURS,
                                communes=COMMUNES_A_SURVEILLER):
    """Signale explicitement si l'ecart date_min/date_max d'un evenement fusionne concernant
    une commune surveillee (Porge, Noisy-sur-Ecole...) depasse seuil_jours - ce qui indiquerait
    que FIREDATE est une date de suivi/mise a jour plutot que la date de depart reelle du feu."""
    for nom in communes:
        concernes = events_fusionnes[events_fusionnes["commune"].str.contains(nom, case=False, na=False)]
        for _, row in concernes.iterrows():
            ecart = (row["date_max"] - row["date_min"]) / pd.Timedelta(days=1)
            if ecart > seuil_jours:
                print(f"  [ALERTE] {row['commune']} : ecart de {ecart:.1f} j entre la premiere "
                      f"({row['date_min'].date().isoformat()}) et la derniere "
                      f"({row['date_max'].date().isoformat()}) date parmi les polygones d'origine. "
                      "FIREDATE semble etre une date de suivi/mise a jour, pas la date de depart du feu.")


def find_events_near(events, lat, lon, rayon_km=5.0):
    """Diagnostic : liste tous les evenements dans un rayon (km) autour d'un point
    (lat, lon), avec leur distance - pour verification manuelle avant fusion."""
    from shapely.geometry import Point

    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(CRS_METRIQUE_FUSION).iloc[0]
    metrique = events.set_geometry("geometry").to_crs(CRS_METRIQUE_FUSION)
    distances = metrique.geometry.distance(point)
    proches = events.loc[distances <= rayon_km * 1000].copy()
    proches["distance_m"] = distances[distances <= rayon_km * 1000].round(0)
    return proches.sort_values("distance_m")


def diagnostiquer_evenement(gdf_brut, commune_pattern, seuil_ha=0):
    """Diagnostic AVANT toute fusion/union : liste tous les polygones EFFIS bruts (un par
    id/OBJECTID EFFIS) dont la commune correspond a `commune_pattern`, avec firedate,
    lastupdate et surface individuelle. Compare aussi AREA_HA (attribut EFFIS) a l'aire
    recalculee depuis la geometrie, pour verifier que le total affiche vient bien de la
    donnee source et pas d'un bug d'union/fusion cote script. A utiliser avant de publier
    un total fusionne suspect (ex. saut brutal de surface d'un run a l'autre)."""
    sel = gdf_brut[gdf_brut["COMMUNE"].str.contains(commune_pattern, case=False, na=False)]
    sel = sel[sel["AREA_HA"] >= seuil_ha].sort_values("FIREDATE").copy()

    sel_m = sel.set_geometry("geometry").to_crs(CRS_METRIQUE_FUSION)
    sel["aire_geom_ha"] = (sel_m.geometry.area / 10000).values

    print(f"  [diagnostic] {len(sel)} polygone(s) EFFIS bruts pour commune~'{commune_pattern}' "
          f"(>= {seuil_ha} ha), AVANT toute fusion :")
    for _, row in sel.iterrows():
        ecart_pct = 100 * (row["AREA_HA"] - row["aire_geom_ha"]) / row["aire_geom_ha"] if row["aire_geom_ha"] else 0
        print(f"      id={row['id']}  {row['COMMUNE']:<20s} firedate={row['FIREDATE']}  "
              f"lastupdate={row['LASTUPDATE']}  AREA_HA={row['AREA_HA']:.1f} ha  "
              f"aire_geom={row['aire_geom_ha']:.1f} ha  (ecart {ecart_pct:+.1f}%)")
    print(f"      TOTAL AREA_HA (somme attribut EFFIS) : {sel['AREA_HA'].sum():.1f} ha")
    print(f"      TOTAL aire geometrique recalculee     : {sel['aire_geom_ha'].sum():.1f} ha")
    return sel


if __name__ == "__main__":
    gdf = fetch_burnt_areas()
    events = to_events(gdf)
    print(f"{len(events)} evenements France recuperes")

    print("--- fusion des evenements proches (espace + temps) ---")
    fusionnes = merge_adjacent_events(events)
    classes = classify_events(fusionnes)
    print(classes["classe"].value_counts())
    print(classes.sort_values("surface_ha", ascending=False).head(10).drop(columns="geometry"))
