"""Recuperation des feux actifs (hotspots NRT) via l'API FIRMS (NASA)."""
import concurrent.futures
import io
import json
import os
import time
from datetime import date as date_cls
from datetime import timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FRANCE_BBOX = (-5.2, 41.3, 9.6, 51.1)  # lon_min, lat_min, lon_max, lat_max
SENSORS = ("VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT")
DEFAULT_MAP_KEY_PATH = Path(__file__).resolve().parent.parent / "map_key.txt"

# L'API FIRMS renvoie parfois un 401/429 transitoire (rate-limit "5000 transactions /
# 10 min" cote NASA, surtout en cas de runs repetes rapproches) plutot qu'une vraie
# cle invalide - on retente avant d'abandonner, comme pour le WFS EFFIS.
MAX_TENTATIVES_FIRMS = 3
DELAI_ENTRE_TENTATIVES_FIRMS_S = 20

# Tolerance de deduplication entre detections proches (meme feu vu par deux
# capteurs/passages differents). Rayon proche de la resolution nominale VIIRS
# (~375 m au nadir, jusqu'a ~800 m en bord de fauchee) : assez petit pour ne
# pas fusionner deux pixels distincts d'un meme front de feu qui s'etend.
DEDUP_RADIUS_M = 500
DEDUP_TIME_WINDOW_MIN = 60

# Filtre Corine Land Cover : requete point par point sur le service ArcGIS REST
# de l'AEE (pas de telechargement local necessaire). On ne garde que les
# classes 3.1.x (forets), 3.2.x (milieux a vegetation arbustive/herbacee) et
# 3.3.x (espaces ouverts peu/pas vegetalises) - objectif = ecarter les sources
# industrielles fixes (1.x tissu urbain/industriel) et agricoles (2.x), pas
# de sur-filtrer les vrais feux de foret/friche.
CLC_IDENTIFY_URL = "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/identify"
CLC_CLASSES_FORET_FRICHE = ("31", "32", "33")
CLC_MAX_WORKERS = 10
CLC_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "clc_cache.json"
CLC_CACHE_DECIMALES = 3  # arrondi lat/lon pour la cle de cache (~110 m de precision)


def _read_map_key(path=DEFAULT_MAP_KEY_PATH):
    """Cle FIRMS : variable d'environnement FIRMS_MAP_KEY en priorite (CI/GitHub Actions,
    ou l'on ne peut pas committer map_key.txt), sinon le fichier local (usage poste local)."""
    env_key = os.environ.get("FIRMS_MAP_KEY")
    if env_key:
        return env_key.strip()
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def fetch_active_fires(bbox=FRANCE_BBOX, days=2, sensors=SENSORS, map_key=None,
                        reference_date=None, timeout=120):
    """Interroge l'API FIRMS area/csv pour chaque capteur et concatene les resultats bruts.

    `reference_date` (date ou None) : derniere date couverte par la requete. None = donnees
    les plus recentes disponibles (comportement par defaut de l'API, "maintenant").
    Sinon interroge la plage [reference_date - (days-1), reference_date] via le parametre
    DATE de l'API FIRMS, ce qui permet de rejouer le script pour une date passee.
    """
    map_key = map_key or _read_map_key()
    bbox_str = ",".join(str(v) for v in bbox)

    start_date = None
    if reference_date is not None:
        if isinstance(reference_date, str):
            reference_date = date_cls.fromisoformat(reference_date)
        start_date = reference_date - timedelta(days=days - 1)

    frames = []
    for sensor in sensors:
        url = f"{FIRMS_BASE_URL}/{map_key}/{sensor}/{bbox_str}/{days}"
        if start_date is not None:
            url += f"/{start_date.isoformat()}"

        derniere_erreur = None
        r = None
        for tentative in range(1, MAX_TENTATIVES_FIRMS + 1):
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                derniere_erreur = e
                print(f"  [firms] {sensor} tentative {tentative}/{MAX_TENTATIVES_FIRMS} echouee ({e!r})")
                if tentative < MAX_TENTATIVES_FIRMS:
                    time.sleep(DELAI_ENTRE_TENTATIVES_FIRMS_S)
        else:
            raise RuntimeError(
                f"Echec de recuperation FIRMS ({sensor}) apres {MAX_TENTATIVES_FIRMS} tentatives"
            ) from derniere_erreur

        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            continue
        df["capteur"] = sensor
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=[
            "latitude", "longitude", "acq_date", "acq_time", "confidence", "capteur", "acq_datetime"
        ])

    all_df = pd.concat(frames, ignore_index=True)
    all_df["acq_datetime"] = pd.to_datetime(
        all_df["acq_date"] + " " + all_df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
    )
    return all_df


def filter_confidence(df):
    """Ecarte les detections a faible confiance (faux positifs probables) :
    VIIRS = confidence categorielle ('l'/'n'/'h'), on ecarte 'l' (low).
    MODIS = confidence numerique 0-100, on ecarte < 50.
    """
    if df.empty:
        return df

    conf = df["confidence"].astype(str).str.strip().str.lower()
    conf_numeric = pd.to_numeric(conf, errors="coerce")

    est_categoriel = conf.isin(["l", "n", "h"])
    garder = pd.Series(True, index=df.index)
    garder[est_categoriel] = conf[est_categoriel] != "l"
    garder[~est_categoriel & conf_numeric.notna()] = conf_numeric[~est_categoriel & conf_numeric.notna()] >= 50

    avant = len(df)
    out = df[garder].reset_index(drop=True)
    print(f"  [filtre confidence] {avant} -> {len(out)} ({avant - len(out)} faux positifs faible confiance ecartes)")
    return out


def dedupe(df, radius_m=DEDUP_RADIUS_M, time_window_min=DEDUP_TIME_WINDOW_MIN):
    """Fusionne les detections tres proches en lat/lon ET en heure (meme feu vu par
    plusieurs capteurs/passages). Deux detections ne sont regroupees que si elles sont
    a la fois a moins de `radius_m` metres ET a moins de `time_window_min` minutes
    l'une de l'autre - une simple grille spatiale fusionnerait a tort des pixels
    distincts d'un meme front de feu observes lors du meme passage satellite."""
    if df.empty:
        return df

    df = df.reset_index(drop=True).copy()

    # distances en metres -> projection metrique (Lambert-93)
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    ).to_crs(2154)
    coords = np.column_stack([pts.geometry.x.values, pts.geometry.y.values])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=radius_m)

    times_min = df["acq_datetime"].values.astype("datetime64[m]").astype(np.int64)

    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    capteur = df["capteur"].values
    for i, j in pairs:
        # ne fusionner que des detections de capteurs differents : deux pixels
        # voisins du meme capteur/passage sont deux points reels d'un front de
        # feu, pas un doublon.
        if capteur[i] == capteur[j]:
            continue
        if abs(int(times_min[i]) - int(times_min[j])) <= time_window_min:
            union(i, j)

    priority = {"h": 2, "n": 1, "l": 0}
    conf_rank = df["confidence"].astype(str).str.lower().map(priority).fillna(-1).values

    clusters = {}
    for idx in range(len(df)):
        clusters.setdefault(find(idx), []).append(idx)

    # metadata de cluster (nb de detections brutes fusionnees, derniere detection, FRP
    # max/moyen) - conservees sur la ligne survivante pour reutilisation en aval (ex.
    # labels_points_actifs.csv, agregation par zone dans build_zoom_bassin_arcachon.py).
    # frp (Fire Radiative Power, MW) est sinon perdu : dedupe() ne garde qu'une ligne par
    # cluster, donc les frp des detections fusionnees doivent etre agreges ici.
    frp_num = pd.to_numeric(df["frp"], errors="coerce")

    n_detections = {}
    derniere_detection = {}
    frp_max = {}
    frp_moyen = {}
    keep_idx = []
    for members in clusters.values():
        best = max(members, key=lambda i: conf_rank[i])
        keep_idx.append(best)
        n_detections[best] = len(members)
        derniere_detection[best] = df["acq_datetime"].iloc[members].max()
        frp_membres = frp_num.iloc[members]
        frp_max[best] = float(frp_membres.max()) if frp_membres.notna().any() else None
        frp_moyen[best] = float(frp_membres.mean()) if frp_membres.notna().any() else None

    keep_idx = sorted(keep_idx)
    out = df.loc[keep_idx].reset_index(drop=True)
    out["n_detections_cluster"] = [n_detections[i] for i in keep_idx]
    out["derniere_detection"] = [derniere_detection[i] for i in keep_idx]
    out["frp_max"] = [frp_max[i] for i in keep_idx]
    out["frp_moyen"] = [frp_moyen[i] for i in keep_idx]
    print(f"  [dedoublonnage] {len(df)} -> {len(out)} ({len(df) - len(out)} doublons multi-capteurs fusionnes)")
    return out


class ClcQueryError(Exception):
    """Echec reseau/format irrecuperable apres toutes les tentatives sur un point."""


def _load_clc_cache(path=CLC_CACHE_PATH):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_clc_cache(cache, path=CLC_CACHE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def _clc_cache_key(lon, lat, decimales=CLC_CACHE_DECIMALES):
    return f"{round(lat, decimales)},{round(lon, decimales)}"


def _query_clc_code(lon, lat, timeout=15, tentatives=3):
    """Interroge le service identify ArcGIS REST de l'AEE pour un point et renvoie
    le code Corine Land Cover 2018 (Code_18, ex. '112', '312') ou None si le point
    est resolu mais ne tombe sur aucune entite CLC (ex. hors couverture).
    Leve ClcQueryError si toutes les tentatives echouent (reseau/timeout/format) -
    a distinguer d'un None, qui est une reponse valide."""
    delta = 0.01
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "sr": "4326",
        "layers": "0",
        "tolerance": "1",
        "mapExtent": f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}",
        "imageDisplay": "400,400,96",
        "returnGeometry": "false",
        "f": "json",
    }
    derniere_erreur = None
    for tentative in range(tentatives):
        try:
            r = requests.get(CLC_IDENTIFY_URL, params=params, timeout=timeout)
            r.raise_for_status()
            results = r.json().get("results", [])
            return results[0]["attributes"].get("Code_18") if results else None
        except (requests.exceptions.RequestException, ValueError, KeyError, IndexError) as e:
            derniere_erreur = e
            if tentative < tentatives - 1:
                time.sleep(1)
    raise ClcQueryError(f"{type(derniere_erreur).__name__}: {derniere_erreur}")


def filter_landcover(df, max_workers=CLC_MAX_WORKERS, cache_path=CLC_CACHE_PATH):
    """Ne garde que les detections tombant sur les classes CLC foret/milieux semi-naturels
    (3.1.x/3.2.x/3.3.x), en interrogeant le service live de l'AEE point par point.

    Un cache local (`cache_path`, cle = lat/lon arrondis) evite de reinterroger l'API
    pour des coordonnees deja vues lors d'une execution precedente. En cas d'echec
    reseau sur un point (apres retries), le point est conserve par defaut (fail-open)
    et l'erreur est logguee - jamais supprime silencieusement pour une raison technique.
    """
    if df.empty:
        return df

    df = df.reset_index(drop=True).copy()
    cache = _load_clc_cache(cache_path)
    cache_modifie = False

    codes = [None] * len(df)
    erreurs = [None] * len(df)
    a_interroger = []

    for i, row in df.iterrows():
        key = _clc_cache_key(row.longitude, row.latitude)
        if key in cache:
            codes[i] = cache[key]
        else:
            a_interroger.append((i, key, row.longitude, row.latitude))

    print(f"  [Corine Land Cover] {len(df) - len(a_interroger)} points depuis le cache local "
          f"({cache_path.name}), {len(a_interroger)} a interroger en direct")

    if a_interroger:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_query_clc_code, lon, lat): (i, key)
                for i, key, lon, lat in a_interroger
            }
            for future in concurrent.futures.as_completed(futures):
                i, key = futures[future]
                try:
                    code = future.result()
                    codes[i] = code
                    cache[key] = code
                    cache_modifie = True
                except ClcQueryError as e:
                    erreurs[i] = str(e)

    if cache_modifie:
        _save_clc_cache(cache, cache_path)

    df["clc_code"] = codes
    classe = df["clc_code"].astype(str).str[:2]

    echec_reseau = pd.Series([e is not None for e in erreurs], index=df.index)
    sans_donnee = df["clc_code"].isna() & ~echec_reseau
    garder = echec_reseau | sans_donnee | classe.isin(CLC_CLASSES_FORET_FRICHE)

    avant = len(df)
    ecartes = df[~garder]
    out = df[garder].reset_index(drop=True)

    print(f"  [filtre Corine Land Cover] {avant} -> {len(out)} ({avant - len(out)} points hors foret/friche ecartes)")
    if echec_reseau.sum():
        print(f"    {echec_reseau.sum()} points en echec reseau CLC, conserves par defaut (fail-open) :")
        for i in df.index[echec_reseau]:
            print(f"      lat={df.loc[i, 'latitude']:.4f} lon={df.loc[i, 'longitude']:.4f} -> {erreurs[i]}")
    if sans_donnee.sum():
        print(f"    {sans_donnee.sum()} points sans donnee CLC (hors couverture), conserves par prudence")
    if not ecartes.empty:
        print("    codes CLC des points ecartes (par frequence) :")
        for code, n in ecartes["clc_code"].value_counts().items():
            print(f"      {code} : {n}")
    return out


def to_geodataframe(df):
    keep = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "capteur", "acq_datetime"]
    for col in ("frp", "clc_code", "n_detections_cluster", "derniere_detection", "frp_max", "frp_moyen"):
        if col in df.columns:
            keep.append(col)
    df = df[keep].copy()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    return gdf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None,
                         help="Date de reference YYYY-MM-DD (dernier jour couvert). "
                              "Insere en dernier segment de l'URL FIRMS area/csv "
                              "(/api/area/csv/[MAP_KEY]/[SOURCE]/[AREA]/[DAY_RANGE]/[DATE]). "
                              "Defaut : donnees les plus recentes (pas de segment DATE).")
    parser.add_argument("--days", type=int, default=2,
                         help="DAY_RANGE FIRMS - nombre de jours couverts, se terminant a --date "
                              "(ou a aujourd'hui si --date est omis). Defaut 2.")
    args = parser.parse_args()

    raw = fetch_active_fires(days=args.days, reference_date=args.date)
    print(f"{len(raw)} detections brutes ({', '.join(SENSORS)})")
    filtre_conf = filter_confidence(raw)
    deduped = dedupe(filtre_conf)
    filtre_clc = filter_landcover(deduped)
    gdf = to_geodataframe(filtre_clc)
    print(gdf.drop(columns="geometry").sort_values("acq_datetime", ascending=False).head(10))
