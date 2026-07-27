"""Reconstitue l'historique des zones actives FIRMS avec une position de slider
PAR PASSAGE SATELLITE (chaque acq_datetime distinct), mais un CONTENU agrege
en FENETRE GLISSANTE 24h : le snapshot au timestamp T inclut toutes les
detections de [T-24h, T], pas seulement celles du passage exact a T. Objectif :
densifier chaque image (plus de clusters visibles) tout en gardant un cran de
slider par passage reel plutot qu'un decoupage horaire arbitraire.

Un seul fetch FIRMS pour toute la periode (DAY_RANGE=4, DATE=debut de
periode) suivi du pipeline existant (confidence + dedoublonnage + CLC),
UNE FOIS. Les positions de slider (acq_datetime distincts) sont identifiees
sur ce jeu complet - verifie sur les donnees reelles du 22-25/07/2026, les
passages sont distincts a la minute pres (pas de bucketing necessaire pour
les IDENTIFIER). Pour chaque position T, le CONTENU du snapshot est ensuite
recalcule sur la fenetre [T-24h, T], pas sur le seul sous-ensemble T exact.

Pour chaque snapshot : buffer+union sur la fenetre 24h (comme
build_zoom_bassin_arcachon.py), export GeoJSON multi-zones (nb_detections_
cluster/frp_max/frp_moyen/derniere_detection par zone, calcules sur la
fenetre), puis archivage via archive_snapshot.py dans data/snapshots/
{snapshot_id}/ (commit+push automatique a chaque snapshot).

zone_brulee (EFFIS) n'est disponible que pour le jour courant (attachee a
tous les snapshots dont la position de slider tombe le jour courant, EFFIS
n'ayant pas de granularite plus fine) - jamais reconstituable pour un jour
anterieur.

Un seul fetch FIRMS pour toute la periode -> rate-limit FIRMS non sollicite
de facon repetee (contrairement a un appel par jour ou par fenetre).

Lancer une seule fois pour l'historique :
    python scripts/backfill_zone_active.py
"""
import sys
from datetime import date as date_cls
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import archive_snapshot as archive  # noqa: E402
import build_zoom_bassin_arcachon as zoom  # noqa: E402
import firms_fetch  # noqa: E402

DATE_DEBUT = date_cls(2026, 7, 22)
FENETRE_GLISSANTE = timedelta(hours=24)

# Limite de l'API FIRMS area/csv sur DAY_RANGE - constatee empiriquement (400 Bad
# Request au-dela, alors que la doc citait "1-5 jours"). Passe la periode totale
# depuis DATE_DEBUT ; au-dela de cette limite, on ne re-scanne que les derniers
# JOURS_MAX_FIRMS jours (les passages plus anciens sont deja archives, donc
# simplement ignores en doublon par archive_snapshot.py, pas perdus).
JOURS_MAX_FIRMS = 5
SEUIL_ALERTE_COUVERTURE = 0.4  # passage signale si < 40% de la moyenne des memes creneaux

# Enveloppe concave (alpha shape, shapely concave_hull ratio) plutot que buffer+union
# ("fleur de cercles") pour la FORME de chaque zone - le regroupement (quels points
# forment une meme zone) reste base sur BUFFER_ACTIF_M, inchange. 0 = tres concave
# (colle aux points, respecte les trous/vides reels type Bassin d'Arcachon), 1 =
# enveloppe convexe (peut combler des trous a tort). 0.3 teste empiriquement sur le
# 24/07 : forme fidele au nuage de points sans exces de decoupage.
ALPHA_DEFAUT = 0.3


def _bucket_horaire(t):
    """VIIRS repasse en pratique 2x/jour sur ce secteur : un creneau nocturne
    (~01h-03h30) et un creneau diurne (~11h-14h). On classe simplement nuit/jour."""
    return "nuit (00h-06h)" if t.hour < 6 else "jour (06h-24h)"


def verifier_couverture_satellite(raw, jour_a_verifier=None, seuil_alerte=SEUIL_ALERTE_COUVERTURE):
    """Compare, pour chaque passage satellite, le nombre de detections BRUTES (avant
    tout filtre) au volume moyen des passages du MEME creneau horaire (nuit/jour) sur
    les AUTRES jours de la periode - pas une comparaison brute jour a jour, qui serait
    faussee par la tendance de fond (le feu grossit globalement). Signale les passages
    nettement en dessous de cette moyenne : signe probable d'un trou de couverture
    satellite (passage manque, nuages, swath partiel), pas d'un vrai creux du feu."""
    par_passage = raw.groupby("acq_datetime").size().sort_index()
    table = par_passage.reset_index(name="n_brutes")
    table["jour"] = table["acq_datetime"].dt.date
    table["creneau"] = table["acq_datetime"].apply(_bucket_horaire)

    print(f"{'passage':<20} {'jour':<12} {'creneau':<16} {'brutes':>8} {'moy. memes creneaux/autres jours':>34} {'ratio':>6}")
    alertes = []
    for _, row in table.iterrows():
        autres = table[(table["creneau"] == row["creneau"]) & (table["jour"] != row["jour"])]
        if autres.empty:
            moyenne, ratio_str = None, "n/a"
        else:
            moyenne = autres["n_brutes"].mean()
            ratio_str = f"{row['n_brutes'] / moyenne:.2f}" if moyenne else "n/a"

        alerte = moyenne is not None and row["n_brutes"] < seuil_alerte * moyenne
        moyenne_str = f"{moyenne:.1f}" if moyenne is not None else "n/a"
        marqueur = "  <-- ALERTE couverture ?" if alerte else ""
        print(f"{str(row['acq_datetime']):<20} {str(row['jour']):<12} {row['creneau']:<16} "
              f"{row['n_brutes']:>8} {moyenne_str:>34} {ratio_str:>6}{marqueur}")
        if alerte:
            alertes.append((row["acq_datetime"], row["n_brutes"], moyenne))

    print()
    if alertes:
        print(f"ALERTES (volume < {seuil_alerte * 100:.0f}% de la moyenne des memes creneaux/autres jours) :")
        for t, n, moy in alertes:
            print(f"  {t} : {n} detections brutes vs {moy:.0f} en moyenne "
                  "-> probable trou de couverture satellite, PAS necessairement un vrai creux du feu")
    else:
        print("Aucune alerte : aucun passage anormalement bas par rapport aux memes creneaux des autres jours.")

    if jour_a_verifier is not None:
        sous = table[table["jour"] == jour_a_verifier]
        print(f"\n--- Focus {jour_a_verifier.isoformat()} ---")
        print(sous[["acq_datetime", "creneau", "n_brutes"]].to_string(index=False))

    return table, alertes


def main(alpha=ALPHA_DEFAUT):
    aujourdhui = date_cls.today()
    periode_totale = (aujourdhui - DATE_DEBUT).days + 1
    jours_couverts = min(periode_totale, JOURS_MAX_FIRMS)
    print(f"Reconstitution par passage satellite (fenetre glissante 24h, enveloppe alpha={alpha}) : "
          f"{DATE_DEBUT.isoformat()} -> {aujourdhui.isoformat()} "
          f"({jours_couverts} jour(s) interroges en 1 seul fetch)")
    if periode_totale > JOURS_MAX_FIRMS:
        print(f"  Periode totale ({periode_totale} j) > limite FIRMS ({JOURS_MAX_FIRMS} j) : "
              f"seuls les {JOURS_MAX_FIRMS} derniers jours sont re-scannes "
              "(les passages plus anciens sont deja archives).")

    raw = firms_fetch.fetch_active_fires(bbox=zoom.ZOOM_BBOX, days=jours_couverts, reference_date=aujourdhui)
    print(f"  {len(raw)} detections FIRMS brutes sur toute la periode")
    if raw.empty:
        print("  Aucune detection sur la periode, rien a reconstituer.")
        return

    filtre_conf = firms_fetch.filter_confidence(raw)
    deduped = firms_fetch.dedupe(filtre_conf)
    filtre_clc = firms_fetch.filter_landcover(deduped)
    if filtre_clc.empty:
        print("  Plus aucune detection apres filtres, rien a reconstituer.")
        return

    passages = sorted(filtre_clc["acq_datetime"].unique())
    print(f"  {len(passages)} position(s) de slider identifiee(s) (un cran par acq_datetime distinct)")

    # output/ n'est pas versionne (jamais commite sur le repo public) - sur un poste
    # local il preexiste souvent d'un run precedent, mais sur un runner CI fraichement
    # clone, le dossier n'existe pas encore.
    zoom.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zone_active_path = zoom.OUTPUT_DIR / "zone_active.geojson"
    zone_active_points_path = zoom.OUTPUT_DIR / "zone_active_points.geojson"
    zone_brulee_path = zoom.OUTPUT_DIR / "zone_brulee.geojson"

    for i, passage in enumerate(passages):
        fetch_timestamp = passage.to_pydatetime()
        print(f"\n=== Snapshot {fetch_timestamp.isoformat()} ({i + 1}/{len(passages)}) "
              f"- fenetre [{(fetch_timestamp - FENETRE_GLISSANTE).isoformat()}, {fetch_timestamp.isoformat()}] ===")

        sous_ensemble = filtre_clc[
            (filtre_clc["acq_datetime"] >= passage - FENETRE_GLISSANTE)
            & (filtre_clc["acq_datetime"] <= passage)
        ]
        gdf = firms_fetch.to_geodataframe(sous_ensemble).to_crs(zoom.CRS_METRIQUE)

        composantes = zoom.construire_composantes(gdf, buffer_m=zoom.BUFFER_ACTIF_M, alpha=alpha)
        print(f"  {len(gdf)} points (fenetre 24h) -> {len(composantes)} composante(s) "
              f"(enveloppe alpha={alpha})")

        zoom._ecrire_composantes_geojson(composantes, fetch_timestamp, zone_active_path)

        # points bruts individuels (lat/lon/frp), sans agregation en zones/hulls - en
        # parallele du polygone, pour un eventuel rendu heatmap cote Lovable. Le
        # polygone n'est pas supprime tant que ce rendu n'est pas valide.
        zoom.exporter_points_actifs_geojson(gdf, fetch_timestamp, zone_active_points_path)

        # zone_brulee (EFFIS) uniquement disponible pour le jour courant - jamais
        # reconstituable pour un jour passe (voir docstring du module).
        jour_du_passage = fetch_timestamp.date()
        zone_brulee_du_passage = zone_brulee_path if jour_du_passage == aujourdhui else None

        entree = archive.archiver(
            zone_active_path=zone_active_path,
            zone_active_points_path=zone_active_points_path,
            zone_brulee_path=zone_brulee_du_passage,
            date_representee=jour_du_passage.isoformat(),
        )
        if entree is not None:
            archive.commit_et_push(entree)

    print(f"\nReconstitution terminee : {len(passages)} passage(s) traite(s).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-couverture", action="store_true", dest="verifier_couverture",
                         help="N'execute que le diagnostic de couverture satellite "
                              "(compare chaque passage aux memes creneaux des autres jours), "
                              "sans regenerer les snapshots.")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAUT,
                         help=f"Ratio shapely concave_hull pour la forme des zones actives "
                              f"(0=tres concave/colle aux points, 1=enveloppe convexe). "
                              f"Defaut {ALPHA_DEFAUT}, teste empiriquement sur le 24/07. "
                              f"Passer une valeur negative pour revenir a l'ancien buffer+union.")
    args = parser.parse_args()
    alpha = None if args.alpha < 0 else args.alpha

    if args.verifier_couverture:
        _aujourdhui = date_cls.today()
        _jours_couverts = min((_aujourdhui - DATE_DEBUT).days + 1, JOURS_MAX_FIRMS)
        _raw = firms_fetch.fetch_active_fires(bbox=zoom.ZOOM_BBOX, days=_jours_couverts, reference_date=_aujourdhui)
        verifier_couverture_satellite(_raw, jour_a_verifier=date_cls(2026, 7, 24))
    else:
        main(alpha=alpha)
