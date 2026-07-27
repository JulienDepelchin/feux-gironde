"""Archive un instantane (snapshot) des donnees du zoom Bassin d'Arcachon dans
data/snapshots/, met a jour le manifest.json, puis commit+push automatiquement
vers le remote 'origin' du repo GitHub public feux-gironde.

Ce repo sert de source de donnees a une appli Lovable (carte interactive +
slider temporel de la progression du feu) qui lit les fichiers bruts GitHub
directement, sans base de donnees - d'ou une structure JSON/GeoJSON simple,
des chemins explicites, et un manifest append-only.

Granularite : UN SNAPSHOT PAR PASSAGE SATELLITE (chaque acq_datetime distinct
dans les donnees FIRMS), pas par jour calendaire - un jour peut compter
plusieurs passages (2 a 6 en pratique sur 22-25/07/2026, cf. backfill_zone_active.py).
zone_active.geojson est une FeatureCollection avec UNE Feature PAR ZONE/CLUSTER
spatial disjoint de ce passage (nb_detections_cluster, frp_max, frp_moyen,
derniere_detection par zone - voir build_zoom_bassin_arcachon.exporter_zone_active_geojson).

A lancer apres chaque nouveau run de build_zoom_bassin_arcachon.py (cas normal,
zone_brulee + zone_active du jour courant) :
    python scripts/build_zoom_bassin_arcachon.py
    python scripts/archive_snapshot.py

Peut aussi etre appele avec des chemins/date explicites pour reconstituer un
passage passe (zone_active seule - voir backfill_zone_active.py) : zone_brulee
(EFFIS) n'est jamais reconstituable pour un jour anterieur au jour courant,
donc `zone_brulee_geojson`/`surface_brulee_ha` valent null dans le manifest
pour ces entrees-la.

Structure produite :
    data/snapshots/{snapshot_id}/zone_brulee.geojson          (absent si non disponible)
    data/snapshots/{snapshot_id}/zone_active.geojson          (FeatureCollection, 1+ zones, Polygon)
    data/snapshots/{snapshot_id}/zone_active_points.geojson   (detections individuelles, Point,
        sans agregation en zones/hulls - pour un eventuel rendu heatmap. Garde en PARALLELE du
        polygone tant que ce rendu n'est pas valide cote Lovable, pas un remplacement.)
    data/snapshots/manifest.json (liste triee par fetch_timestamp, un objet par passage)

ATTENTION : commit + push automatiques a chaque run (choix assume - pas de
validation manuelle avant publication sur le repo public). Verifier la
connexion VPN pro avant de lancer ce script si le reseau l'exige.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"
MANIFEST_PATH = SNAPSHOTS_DIR / "manifest.json"

SOURCE_LABEL = "EFFIS modis.ba.poly.season + FIRMS VIIRS NOAA20/SNPP"


def _snapshot_id(fetch_timestamp_iso):
    """Identifiant de dossier a partir du fetch_timestamp (ISO) - lisible, trie
    chronologiquement par simple tri alphabetique, unique par passage satellite."""
    return fetch_timestamp_iso.replace(":", "-").split(".")[0]


def archiver(zone_active_path=None, zone_active_points_path=None, zone_brulee_path=None, date_representee=None):
    """Archive un snapshot pour un passage satellite donne (zone_active.geojson =
    FeatureCollection avec une Feature par zone/cluster spatial).

    zone_active_path : GeoJSON zone active - polygones agreges (obligatoire - toujours
        reconstituable, y compris pour un jour passe via l'API FIRMS avec DATE explicite).
    zone_active_points_path : GeoJSON points individuels bruts, sans agregation (optionnel -
        garde en parallele du polygone pour un eventuel rendu heatmap ; si absent, simplement
        pas archive pour ce snapshot, le polygone reste la seule sortie).
    zone_brulee_path : GeoJSON zone brulee (optionnel - None pour un passage anterieur
        au jour courant, cf. limite EFFIS documentee dans le docstring du module). Si
        fourni mais que son fetch_timestamp ne correspond pas au jour represente, il
        est ignore.
    date_representee : jour calendaire (str "YYYY-MM-DD") du passage - sert uniquement
        a decider si zone_brulee s'applique. Par defaut, deduit du fetch_timestamp de
        zone_active.

    Renvoie None si ce passage (meme snapshot_id) est deja present dans le manifest.
    """
    zone_active_path = Path(zone_active_path) if zone_active_path else (OUTPUT_DIR / "zone_active.geojson")
    zone_active_points_path = Path(zone_active_points_path) if zone_active_points_path else None
    zone_brulee_path = Path(zone_brulee_path) if zone_brulee_path else None

    if not zone_active_path.exists():
        raise FileNotFoundError(
            f"{zone_active_path} introuvable - lancer le fetch des zones actives avant archive_snapshot.py"
        )

    zone_active = json.loads(zone_active_path.read_text(encoding="utf-8"))
    features = zone_active["features"]
    if not features:
        raise ValueError(f"{zone_active_path} ne contient aucune Feature - rien a archiver")

    # meme fetch_timestamp pour toutes les Features d'un passage (voir
    # exporter_zone_active_geojson) : on le lit sur la premiere.
    fetch_timestamp = features[0]["properties"]["fetch_timestamp"]
    date_representee = date_representee or fetch_timestamp[:10]
    snapshot_id = _snapshot_id(fetch_timestamp)

    nb_zones_actives = len(features)
    nb_detections_total = sum(f["properties"].get("nb_detections_cluster", 0) or 0 for f in features)
    derniere_detection = max(f["properties"]["derniere_detection"] for f in features)
    frp_values = [f["properties"]["frp_max"] for f in features if f["properties"].get("frp_max") is not None]
    frp_max = max(frp_values) if frp_values else None

    zone_brulee = None
    proprietes_brulee = None
    if zone_brulee_path is not None and zone_brulee_path.exists():
        candidate = json.loads(zone_brulee_path.read_text(encoding="utf-8"))
        candidate_props = candidate["features"][0]["properties"]
        if candidate_props["fetch_timestamp"][:10] == date_representee:
            zone_brulee, proprietes_brulee = candidate, candidate_props
        else:
            print(f"  zone_brulee ignoree pour {date_representee} : son fetch_timestamp "
                  f"({candidate_props['fetch_timestamp']}) ne correspond pas a ce jour.")

    manifest = {"snapshots": []}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    # migration retro-compatible : les tout premiers snapshots (avant l'ajout du champ
    # `date` pour le backfill) n'avaient qu'un fetch_timestamp - on en deduit `date`.
    for s in manifest["snapshots"]:
        if "date" not in s:
            s["date"] = s["fetch_timestamp"][:10]

    if any(s["snapshot_id"] == snapshot_id for s in manifest["snapshots"]):
        print(f"Snapshot {snapshot_id} deja present dans le manifest, rien a archiver.")
        return None

    dossier = SNAPSHOTS_DIR / snapshot_id
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "zone_active.geojson").write_text(
        json.dumps(zone_active, ensure_ascii=False), encoding="utf-8"
    )

    entree = {
        "date": date_representee,
        "fetch_timestamp": fetch_timestamp,
        "snapshot_id": snapshot_id,
        "zone_brulee_geojson": None,
        "zone_active_geojson": f"data/snapshots/{snapshot_id}/zone_active.geojson",
        "zone_active_points_geojson": None,
        "surface_brulee_ha": None,
        "nb_zones_actives": nb_zones_actives,
        "nb_detections_total": nb_detections_total,
        "frp_max": frp_max,
        "derniere_detection": derniere_detection,
        "source": SOURCE_LABEL,
    }

    if zone_active_points_path is not None and zone_active_points_path.exists():
        (dossier / "zone_active_points.geojson").write_text(
            zone_active_points_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        entree["zone_active_points_geojson"] = f"data/snapshots/{snapshot_id}/zone_active_points.geojson"

    if zone_brulee is not None:
        (dossier / "zone_brulee.geojson").write_text(
            json.dumps(zone_brulee, ensure_ascii=False), encoding="utf-8"
        )
        entree["zone_brulee_geojson"] = f"data/snapshots/{snapshot_id}/zone_brulee.geojson"
        entree["surface_brulee_ha"] = proprietes_brulee["surface_ha"]

    manifest["snapshots"].append(entree)
    manifest["snapshots"].sort(key=lambda s: s["fetch_timestamp"])
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Snapshot archive ({snapshot_id}) : {dossier}")
    print(f"Manifest mis a jour : {MANIFEST_PATH} ({len(manifest['snapshots'])} snapshot(s) au total)")
    return entree


def commit_et_push(entree):
    subprocess.run(["git", "add", "data/snapshots/"], cwd=ROOT, check=True)

    surface = f"{entree['surface_brulee_ha']:.0f} ha brules" if entree["surface_brulee_ha"] is not None \
        else "zone brulee non disponible"
    message = (f"Ajout snapshot {entree['snapshot_id']} : {surface}, "
               f"{entree['nb_zones_actives']} zones actives, {entree['nb_detections_total']} detections")

    resultat = subprocess.run(
        ["git", "commit", "-m", message], cwd=ROOT, capture_output=True, text=True
    )
    if resultat.returncode != 0:
        if "nothing to commit" in resultat.stdout.lower():
            print("Rien a committer (snapshot deja versionne).")
            return
        raise RuntimeError(f"Echec du commit :\n{resultat.stdout}\n{resultat.stderr}")
    print(resultat.stdout.strip())

    subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
    print("Push effectue vers origin/main.")


if __name__ == "__main__":
    entree_ajoutee = archiver()
    if entree_ajoutee is not None:
        commit_et_push(entree_ajoutee)
